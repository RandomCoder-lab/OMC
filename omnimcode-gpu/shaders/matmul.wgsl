// Naive WGSL matmul kernel.
//
// One thread per output cell. C[i, j] = sum_k A[i, k] * B[k, j].
// 16x16 workgroup = 256 threads per dispatch group (typical "warp"-
// friendly size for Polaris and most GPUs we'd target).
//
// Shape passed via uniform buffer (m, k, n, _pad) so we can dispatch
// the same shader against any input dimensions.
//
// Performance note: this is the "ground truth" GPU kernel — no
// tiling, no shared-memory caching. For real adoption we'd add a
// tiled variant. The point of v0.7 is wiring; perf is v0.8.

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
    for (var kk: u32 = 0u; kk < shape.k; kk = kk + 1u) {
        acc = acc + a[i * shape.k + kk] * b[kk * shape.n + j];
    }
    c[i * shape.n + j] = acc;
}
