// Parameterized WGSL matmul kernel — workgroup tile size + inner-loop
// body are substituted at module load time by `WgpuBackend::new_async`.
//
// The default substitution gives the standard 16×16 linear-K accumulator
// (one thread per output cell). Other tiles (13×13, 21×21, 8×32, ...) and
// the substrate Fib-K-stride variant come from the same template.
//
// Shape is passed via a small uniform buffer so we can dispatch this
// shader against any input dimensions.

struct Shape {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let j: u32 = gid.y;
    if (i >= shape.m || j >= shape.n) {
        return;
    }
    var acc: f32 = 0.0;
    // __INNER_LOOP__
    c[i * shape.n + j] = acc;
}
