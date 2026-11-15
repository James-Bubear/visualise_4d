struct Uniforms {
    camera_pos: vec4<f32>,
    camera_basis: mat4x4<f32>,
    view_normal: vec4<f32>,
    proj_params: vec4<f32>,
    time_and_flags: vec4<f32>,
    resolution: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

fn vec4_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a + b; }
fn vec4_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a - b; }
fn vec4_scale(a: vec4<f32>, s: f32) -> vec4<f32> { return a * s; }
fn vec4_len(a: vec4<f32>) -> f32 { return length(a); }

fn vec4_norm(a: vec4<f32>) -> vec4<f32> {
    let l = vec4_len(a);
    if l == 0.0 { return vec4<f32>(0.0); }
    return a / l;
}

fn sd_sphere(p: vec4<f32>, r: f32) -> f32 {
    return vec4_len(p) - r;
}

fn sd_box(p: vec4<f32>, b: vec4<f32>) -> f32 {
    let q = abs(p) - b;
    let max_q = max(max(q.x, q.y), max(q.z, q.w));
    let pos_q = max(q, vec4<f32>(0.0));
    return length(pos_q) + min(max_q, 0.0);
}

// Scene:
//   - Hypersphere: radius 1.5, centered at the origin.
//   - Tesseract:   half-extent 1.0 on each axis, centered at [3, 0, 0, 0].
// These values deliberately match src/scene.rs so the CPU fallback
// describes the same world the GPU renders.
//
// NOTE: the previous version had both objects concentric with the sphere
// (r = 3) exactly inscribed in the tesseract (half-extent 3), so the box's
// distance was <= the sphere's everywhere and the sphere was never visible.
//
// Returns vec2(distance, object_id): 0 = nothing, 1 = sphere, 2 = tesseract.
fn map(p: vec4<f32>) -> vec2<f32> {
    var best_dist = 1000.0;
    var best_id = 0.0;

    if uniforms.time_and_flags.y > 0.5 {
        let d = sd_sphere(p, 1.5);
        if d < best_dist { best_dist = d; best_id = 1.0; }
    }

    if uniforms.time_and_flags.z > 0.5 {
        let p_offset = vec4_sub(p, vec4<f32>(3.0, 0.0, 0.0, 0.0));
        let d = sd_box(p_offset, vec4<f32>(1.0));
        if d < best_dist { best_dist = d; best_id = 2.0; }
    }

    return vec2<f32>(best_dist, best_id);
}

// Proper 4D surface normal: central-difference gradient of the distance
// field along all FOUR axes (8 samples). A 4D normal must have nonzero
// z/w components in general; zeroing them out (as the old CPU version did)
// produces a gradient that is not perpendicular to the hit surface.
fn calc_normal(p: vec4<f32>) -> vec4<f32> {
    let h = 0.001;
    let d0 = map(vec4_add(p, vec4<f32>(h, 0.0, 0.0, 0.0))).x;
    let d1 = map(vec4_add(p, vec4<f32>(-h, 0.0, 0.0, 0.0))).x;
    let d2 = map(vec4_add(p, vec4<f32>(0.0, h, 0.0, 0.0))).x;
    let d3 = map(vec4_add(p, vec4<f32>(0.0, -h, 0.0, 0.0))).x;
    let d4 = map(vec4_add(p, vec4<f32>(0.0, 0.0, h, 0.0))).x;
    let d5 = map(vec4_add(p, vec4<f32>(0.0, 0.0, -h, 0.0))).x;
    let d6 = map(vec4_add(p, vec4<f32>(0.0, 0.0, 0.0, h))).x;
    let d7 = map(vec4_add(p, vec4<f32>(0.0, 0.0, 0.0, -h))).x;
    return vec4_norm(vec4<f32>(
        (d0 - d1) / (2.0 * h),
        (d2 - d3) / (2.0 * h),
        (d4 - d5) / (2.0 * h),
        (d6 - d7) / (2.0 * h)
    ));
}

// Adaptive sphere tracing. Returns vec2(t, object_id);
// on a miss returns vec2(-1.0, 0.0).
fn ray_march(ro: vec4<f32>, rd: vec4<f32>) -> vec2<f32> {
    var t = 0.01;
    var max_steps = 200;
    var min_step = 0.0005;

    for (var i = 0; i < max_steps; i++) {
        let p = vec4_add(ro, vec4_scale(rd, t));
        let res = map(p);
        let d = res.x;
        let id = res.y;

        if d < 0.005 {
            return vec2<f32>(t, id);
        }

        // Adaptive step size.
        var step_size = d;
        if (d > 1.0) {
            step_size = d * 0.5; // Don't jump too far when far away
        } else if (d < 0.5) {
            step_size = max(d * 0.7, min_step); // Slow down drastically when close
        }

        t += step_size;

        if t > 100.0 { break; }
    }
    return vec2<f32>(-1.0, 0.0);
}

// Symmetric 4-point perspective: the constant term of the ray direction is
// spread equally over all four axes (each gets +1/4), so the center-of-screen
// gaze is the normalized diagonal (1,1,1,1)/2 and Dx+Dy+Dz+Dw = 1 on every
// ray. Screen offsets trade axes against each other pairwise; tanh keeps the
// resulting direction patch bounded on the 3-sphere (soft fisheye).
fn get_4d_direction(uv: vec2<f32>) -> vec4<f32> {
    let zoom = 0.6;
    let uv_zoomed = uv * zoom;
    let scale = 3.0;
    let compressed_uv = vec2<f32>(
        tanh(uv_zoomed.x * scale),
        tanh(uv_zoomed.y * scale)
    );

    let u_c = compressed_uv.x;
    let v_c = compressed_uv.y;

    let inv4 = 0.25;
    let Dx = inv4 * (-u_c + v_c + 1.0);
    let Dy = inv4 * ( u_c + v_c + 1.0);
    let Dz = inv4 * (-u_c - v_c + 1.0);
    let Dw = inv4 * ( u_c - v_c + 1.0);

    let D = vec4<f32>(Dx, Dy, Dz, Dw);
    return vec4_norm(D);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let flipped_y = uniforms.resolution.y - frag_coord.y;
    let uv = (vec2<f32>(frag_coord.x, flipped_y) - uniforms.resolution.xy * 0.5) / uniforms.resolution.y;

    let rd_cam = get_4d_direction(uv);
    let rd_world = uniforms.camera_basis * rd_cam;
    let ro_world = uniforms.camera_pos;

    let res = ray_march(ro_world, rd_world);
    let t = res.x;
    let id = res.y;

    if t < 0.0 { return vec4<f32>(0.04, 0.04, 0.08, 1.0); }

    let p = vec4_add(ro_world, vec4_scale(rd_world, t));
    let n = calc_normal(p);
    let light = vec4_norm(-rd_world);
    let diff = max(dot(n, light), 0.0);

    var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if id == 1.0 { col = vec4<f32>(0.9, 0.5, 0.2, 1.0) * diff; }
    else if id == 2.0 { col = vec4<f32>(0.4, 0.3, 0.9, 1.0) * diff; }

    return col + vec4<f32>(0.15, 0.15, 0.15, 0.0);
}
