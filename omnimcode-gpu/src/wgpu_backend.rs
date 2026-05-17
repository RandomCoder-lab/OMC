//! wgpu (Vulkan / Metal / DX12 / OpenGL compute) backend.
//!
//! Cross-vendor GPU compute via the `wgpu` crate. The safe default
//! for AMD Polaris (RX 580 / gfx803) hardware — it talks to the
//! Vulkan driver without needing ROCm. Also works on NVIDIA, Apple
//! Silicon (Metal), and Windows (DX12) with the same kernel.
//!
//! Trade-off: portability over raw FLOPS. A tuned cuBLAS or rocBLAS
//! kernel will beat this; the point of the v0.7 scaffold is to have
//! a working GPU path that won't crash anyone's machine.
//!
//! ## Setup overhead
//!
//! `WgpuBackend::new` does the one-time device + queue + pipeline
//! creation (~10s of ms). Reuse a single instance for many matmuls;
//! don't construct one per call.
//!
//! ## How the kernel runs
//!
//! 1. Upload A, B, and a small uniform buffer with the shape ints
//! 2. Allocate the C output buffer
//! 3. Dispatch `ceil(m/16) × ceil(n/16) × 1` workgroups of 16×16 threads
//! 4. Submit + poll
//! 5. Copy C back into host memory

use bytemuck::{Pod, Zeroable};

use crate::{BackendError, ComputeBackend, Matrix};

/// Standard linear-K accumulator body. Plain `for k in 0..K: acc += a[k]*b[k]`.
const LINEAR_K_BODY: &str = "\
    for (var kk: u32 = 0u; kk < shape.k; kk = kk + 1u) {
        acc = acc + a[i * shape.k + kk] * b[kk * shape.n + j];
    }
";

/// Substrate-native Fibonacci K-stride accumulator. Walks K in chunks of
/// Fibonacci sizes (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377,
/// 610, 987, 1597, 2584, 4181, 6765) so each partial sum spans a
/// substrate-shaped slice of the reduction. Mathematically equivalent to
/// linear (f32 sum order differs but precision impact is tiny). The
/// hypothesis: substrate-chunked accumulation may hit cache line / wavefront
/// geometry differently than linear-K and could be faster (or could lose).
const FIB_K_STRIDE_BODY: &str = "\
    // Fibonacci sequence up to a useful K bound. The array literal is the
    // substrate's first 20 attractors (excluding the 0 leaf).
    var fibs = array<u32, 20>(
        1u, 1u, 2u, 3u, 5u, 8u, 13u, 21u, 34u, 55u,
        89u, 144u, 233u, 377u, 610u, 987u, 1597u, 2584u, 4181u, 6765u
    );
    var pos: u32 = 0u;
    var fi: u32 = 0u;
    loop {
        if (pos >= shape.k) { break; }
        var chunk: u32 = fibs[fi];
        if (pos + chunk > shape.k) { chunk = shape.k - pos; }
        // Inner per-chunk accumulator — partial sum over the Fib chunk.
        var part: f32 = 0.0;
        for (var kk: u32 = pos; kk < pos + chunk; kk = kk + 1u) {
            part = part + a[i * shape.k + kk] * b[kk * shape.n + j];
        }
        acc = acc + part;
        pos = pos + chunk;
        // Cycle through the Fib table; once we've used the largest, restart.
        // For typical K (256-1024) we'll never exceed index 16.
        fi = fi + 1u;
        if (fi >= 20u) { fi = 0u; }
    }
";

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ShapeUniform {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Which kernel variant to compile. v0.8.3 substrate-K-stride explores
/// whether chunking the inner K accumulation in Fibonacci-sized blocks
/// (1, 1, 2, 3, 5, 8, 13, 21, ...) — which match L1 cache-line geometry
/// at certain points — improves matmul throughput vs the standard
/// linear-K accumulation.
#[derive(Copy, Clone, Debug)]
pub enum MatmulKernel {
    /// Standard `acc += a[i,k]*b[k,j]` for k in 0..K. The everywhere default.
    Linear,
    /// Substrate-native: walk K in Fibonacci-sized chunks. Equivalent
    /// math (sum order differs slightly), different memory access pattern.
    FibKStride,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Workgroup tile (tile_x × tile_y). 16×16 is the conventional default;
    /// Fibonacci tiles (13×13, 21×21) + anisotropic shapes (8×32, 32×8)
    /// are v0.8.3's substrate-native variants.
    tile_x: u32,
    tile_y: u32,
    /// Active kernel variant — linear K accumulation (default) or
    /// substrate Fib-K-stride.
    kernel: MatmulKernel,
    /// Adapter info for diagnostics — backend name, vendor, device.
    pub adapter_info: wgpu::AdapterInfo,
}

impl WgpuBackend {
    /// Initialize the wgpu device + compile the matmul kernel with
    /// the standard 16×16 workgroup and linear-K accumulation. Blocking.
    pub fn new() -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async(16, 16, MatmulKernel::Linear))
    }

    /// Square-tile constructor (NxN). Equivalent to `with_tile_xy(N, N)`.
    pub fn with_tile(tile: u32) -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async(tile, tile, MatmulKernel::Linear))
    }

    /// Anisotropic tile constructor (tx × ty). 8×32 / 32×8 etc.
    pub fn with_tile_xy(tx: u32, ty: u32) -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async(tx, ty, MatmulKernel::Linear))
    }

    /// Full constructor — pick tile shape AND kernel variant.
    /// `MatmulKernel::FibKStride` walks the inner K loop in Fibonacci-
    /// sized chunks; the rest of the surrounding scaffolding is identical.
    pub fn with_config(tx: u32, ty: u32, kernel: MatmulKernel) -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async(tx, ty, kernel))
    }

    async fn new_async(tile_x: u32, tile_y: u32, kernel: MatmulKernel) -> Result<Self, BackendError> {
        if tile_x == 0 || tile_y == 0 {
            return Err(BackendError::Backend("tile dims must be > 0".to_string()));
        }
        let tile = (tile_x.max(tile_y)) as u32;
        let _ = tile;  // (used for limit calc below as tx*ty)
        // BackendOptions::all() opens Vulkan on Linux/Windows, Metal
        // on macOS, DX12 on Windows. On RX 580 specifically: Vulkan.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or_else(|| BackendError::Backend(
            "no compatible GPU adapter found — install vulkan-driver \
             (or equivalent) and try again".to_string()
        ))?;
        let adapter_info = adapter.get_info();
        // Pick limits: keep downlevel defaults when tile²≤256, otherwise
        // request bigger workgroup-invocation limits so 21×21=441 and
        // friends can be created. Polaris/Vulkan typically allows up to
        // 1024 invocations per workgroup, so 13×13 / 21×21 are fine,
        // 34×34=1156 is past the line on this hardware.
        let need = tile_x * tile_y;
        let mut limits = wgpu::Limits::downlevel_defaults();
        if need > limits.max_compute_invocations_per_workgroup {
            limits.max_compute_invocations_per_workgroup = need;
        }
        if tile_x > limits.max_compute_workgroup_size_x {
            limits.max_compute_workgroup_size_x = tile_x;
        }
        if tile_y > limits.max_compute_workgroup_size_y {
            limits.max_compute_workgroup_size_y = tile_y;
        }
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("omc-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ).await.map_err(|e| BackendError::Backend(format!(
            "request_device (tile={}x{}): {}", tile_x, tile_y, e
        )))?;

        // WGSL workgroup_size must be a literal in source. Substitute
        // the tile size into the shader at module-load time, plus pick
        // the inner-loop body (linear K or Fibonacci K-stride).
        let src_template = include_str!("../shaders/matmul.wgsl");
        let src = src_template
            .replace(
                "@workgroup_size(16, 16, 1)",
                &format!("@workgroup_size({}, {}, 1)", tile_x, tile_y),
            )
            .replace(
                "// __INNER_LOOP__",
                match kernel {
                    MatmulKernel::Linear => LINEAR_K_BODY,
                    MatmulKernel::FibKStride => FIB_K_STRIDE_BODY,
                },
            );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul.wgsl"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul-bgl"),
            entries: &[
                // 0: shape uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: A (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: B (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: C (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul-pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "matmul",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self { device, queue, pipeline, bind_group_layout,
                  tile_x, tile_y, kernel, adapter_info })
    }

    /// Returns the workgroup tile shape (tile_x, tile_y) this backend was
    /// created with.
    pub fn tile(&self) -> (u32, u32) { (self.tile_x, self.tile_y) }

    /// Returns the matmul kernel variant this backend was compiled with.
    pub fn kernel(&self) -> MatmulKernel { self.kernel }

    /// Print adapter info — useful for debugging which device the
    /// kernel actually ran on (integrated vs discrete, driver version,
    /// etc.). Run `cargo run --features wgpu --example device_info`
    /// (when we add that example) to dump it.
    pub fn describe_adapter(&self) -> String {
        format!(
            "{} (vendor={}, device={}, type={:?}, backend={:?}, driver={:?})",
            self.adapter_info.name,
            self.adapter_info.vendor,
            self.adapter_info.device,
            self.adapter_info.device_type,
            self.adapter_info.backend,
            self.adapter_info.driver,
        )
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &'static str { "wgpu" }

    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, BackendError> {
        if a.cols != b.rows {
            return Err(BackendError::ShapeMismatch { lhs: a.shape(), rhs: b.shape() });
        }
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let shape = ShapeUniform { m: m as u32, k: k as u32, n: n as u32, _pad: 0 };

        use wgpu::util::DeviceExt;
        let shape_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape"),
            contents: bytemuck::bytes_of(&shape),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let a_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: bytemuck::cast_slice(&a.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B"),
            contents: bytemuck::cast_slice(&b.data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: c_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shape_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul-enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Dispatch ceil(m/tile_x) × ceil(n/tile_y) × 1.
            let gx = (m as u32 + self.tile_x - 1) / self.tile_x;
            let gy = (n as u32 + self.tile_y - 1) / self.tile_y;
            pass.dispatch_workgroups(gx, gy, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &readback_buf, 0, c_size);
        self.queue.submit(Some(encoder.finish()));

        // Map + read back. The poll-wait is unfortunately mandatory
        // because wgpu's buffer mapping is async.
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| BackendError::Backend(format!("readback channel: {}", e)))?
            .map_err(|e| BackendError::Backend(format!("map_async: {}", e)))?;
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback_buf.unmap();
        Ok(Matrix::new(m, n, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    /// Try to construct a wgpu backend. CI machines often lack a GPU;
    /// skip rather than fail if init doesn't succeed.
    fn try_wgpu() -> Option<WgpuBackend> {
        match WgpuBackend::new() {
            Ok(b) => {
                eprintln!("wgpu adapter: {}", b.describe_adapter());
                Some(b)
            }
            Err(e) => {
                eprintln!("wgpu unavailable on this machine ({}); skipping", e);
                None
            }
        }
    }

    #[test]
    fn wgpu_matmul_matches_cpu_8x8() {
        let Some(gpu) = try_wgpu() else { return };
        let a_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..64).map(|i| ((63 - i) as f32) * 0.1).collect();
        let a = Matrix::new(8, 8, a_data);
        let b = Matrix::new(8, 8, b_data);
        let cpu_out = CpuBackend.matmul(&a, &b).unwrap();
        let gpu_out = gpu.matmul(&a, &b).unwrap();
        let diff = cpu_out.max_abs_diff(&gpu_out);
        assert!(diff < 1e-4, "GPU and CPU disagree (max diff {})", diff);
    }

    #[test]
    fn wgpu_matmul_basic_2x3_3x2() {
        let Some(gpu) = try_wgpu() else { return };
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let cpu_out = CpuBackend.matmul(&a, &b).unwrap();
        let gpu_out = gpu.matmul(&a, &b).unwrap();
        let diff = cpu_out.max_abs_diff(&gpu_out);
        assert!(diff < 1e-5, "diff {}", diff);
    }

    #[test]
    fn wgpu_shape_mismatch_errors() {
        let Some(gpu) = try_wgpu() else { return };
        let a = Matrix::new(2, 3, vec![0.0; 6]);
        let b = Matrix::new(4, 2, vec![0.0; 8]);
        assert!(matches!(gpu.matmul(&a, &b), Err(BackendError::ShapeMismatch { .. })));
    }
}
