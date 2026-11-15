use crate::math::*;

// CPU-side mirror of shader.wgsl. Kept for potential CPU fallback /
// offline testing, but kept IN SYNC with the GPU scene:
//   - Hypersphere: radius 1.5, centered at the origin.
//   - Tesseract:   half-extent 1.0 per axis, centered at [3, 0, 0, 0].

pub fn sd_sphere(p: Vec4, r: f32) -> f32 {
    vec4_len(p) - r
}

pub fn sd_box(p: Vec4, b: Vec4) -> f32 {
    let q = [
        p[0].abs() - b[0],
        p[1].abs() - b[1],
        p[2].abs() - b[2],
        p[3].abs() - b[3],
    ];
    let max_q = q.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let pos_q = [
        q[0].max(0.0),
        q[1].max(0.0),
        q[2].max(0.0),
        q[3].max(0.0),
    ];
    vec4_len(pos_q) + max_q.min(0.0)
}

/// Returns (distance, object_id).
/// 0 = nothing, 1 = sphere, 2 = tesseract
pub fn map(p: Vec4, show_sphere: bool, show_tesseract: bool) -> (f32, u8) {
    let mut best_dist = f32::MAX;
    let mut best_id = 0u8;

    if show_sphere {
        let d = sd_sphere(p, 1.5);
        if d < best_dist {
            best_dist = d;
            best_id = 1;
        }
    }

    if show_tesseract {
        let p_offset = vec4_sub(p, [3.0, 0.0, 0.0, 0.0]);
        let d = sd_box(p_offset, [1.0, 1.0, 1.0, 1.0]);
        if d < best_dist {
            best_dist = d;
            best_id = 2;
        }
    }

    (best_dist, best_id)
}

/// Proper 4D surface normal: central-difference gradient of the distance
/// field along ALL FOUR axes (8 samples). The previous version only
/// differentiated along x and y and hardcoded z/w to zero, which produced
/// a "normal" that was not perpendicular to the hit surface.
pub fn calc_normal(p: Vec4, show_sphere: bool, show_tesseract: bool) -> Vec4 {
    let h = 0.001;
    let d0 = map(vec4_add(p, [ h, 0.0, 0.0, 0.0]), show_sphere, show_tesseract).0;
    let d1 = map(vec4_add(p, [-h, 0.0, 0.0, 0.0]), show_sphere, show_tesseract).0;
    let d2 = map(vec4_add(p, [0.0,  h, 0.0, 0.0]), show_sphere, show_tesseract).0;
    let d3 = map(vec4_add(p, [0.0, -h, 0.0, 0.0]), show_sphere, show_tesseract).0;
    let d4 = map(vec4_add(p, [0.0, 0.0,  h, 0.0]), show_sphere, show_tesseract).0;
    let d5 = map(vec4_add(p, [0.0, 0.0, -h, 0.0]), show_sphere, show_tesseract).0;
    let d6 = map(vec4_add(p, [0.0, 0.0, 0.0,  h]), show_sphere, show_tesseract).0;
    let d7 = map(vec4_add(p, [0.0, 0.0, 0.0, -h]), show_sphere, show_tesseract).0;

    vec4_norm([
        (d0 - d1) / (2.0 * h),
        (d2 - d3) / (2.0 * h),
        (d4 - d5) / (2.0 * h),
        (d6 - d7) / (2.0 * h),
    ])
}

/// Adaptive sphere tracing, mirroring shader.wgsl's ray_march.
/// Returns (t, object_id): on a hit, t is the distance along the ray and
/// object_id identifies the surface (1 = sphere, 2 = tesseract).
/// On a miss returns (-1.0, 0).
pub fn ray_march(ro: Vec4, rd: Vec4, show_sphere: bool, show_tesseract: bool) -> (f32, u8) {
    let mut t = 0.01;
    let max_steps = 200;
    let min_step = 0.0005;

    for _ in 0..max_steps {
        let p = vec4_add(ro, vec4_scale(rd, t));
        let (d, id) = map(p, show_sphere, show_tesseract);

        if d < 0.005 {
            return (t, id);
        }

        // Adaptive step size (same policy as the shader).
        let step_size = if d > 1.0 {
            d * 0.5
        } else if d < 0.5 {
            (d * 0.7).max(min_step)
        } else {
            d
        };

        t += step_size;

        if t > 100.0 {
            break;
        }
    }
    (-1.0, 0)
}
