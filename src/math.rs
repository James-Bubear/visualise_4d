pub type Vec4 = [f32; 4];
pub type Mat4 = [[f32; 4]; 4];

pub fn mat4_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut res = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[i][k] * b[k][j];
            }
            res[i][j] = sum;
        }
    }
    res
}

pub fn mat4_transpose(m: &Mat4) -> Mat4 {
    let mut res = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            res[i][j] = m[j][i];
        }
    }
    res
}

pub fn vec4_dot(a: Vec4, b: Vec4) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

pub fn vec4_add(a: Vec4, b: Vec4) -> Vec4 {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
}

pub fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]
}

pub fn vec4_scale(a: Vec4, s: f32) -> Vec4 {
    [a[0]*s, a[1]*s, a[2]*s, a[3]*s]
}

pub fn vec4_len(a: Vec4) -> f32 {
    (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]).sqrt()
}

pub fn vec4_norm(a: Vec4) -> Vec4 {
    let l = vec4_len(a);
    if l == 0.0 { [0.0, 0.0, 0.0, 0.0] } else { vec4_scale(a, 1.0/l) }
}

pub fn rot4(i: usize, j: usize, theta: f32) -> Mat4 {
    let c = theta.cos();
    let s = theta.sin();
    let mut m = mat4_identity();
    m[i][i] = c; m[i][j] = -s;
    m[j][i] = s; m[j][j] = c;
    m
}