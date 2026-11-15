# 4D Raymarcher

> [!NOTE]
> **A note on how this was made:** This project was built as an AI-assisted experiment — a back-and-forth between me and [Lumo](https://proton.me/lumo), Proton's AI assistant (a style of development often called "vibe coding" or AI-assisted development). I directed the project, made the design decisions, and drove the debugging; Lumo wrote the code and this README.

A real-time raymarcher that renders scenes in **four spatial dimensions**, built in Rust with [wgpu](https://wgpu.rs/) and [winit](https://crates.io/crates/winit). Scenes consist of a hypersphere and a tesseract. Each pixel is shaded by marching rays through a 4D signed distance field on the GPU, in a single WGSL fragment shader.

Instead of a conventional pinhole projection, this renderer uses a **symmetric "4-point perspective"**: the camera's forward direction is the diagonal `(1, 1, 1, 1)/√4`, spreading the gaze evenly across all four axes rather than favoring one "depth" axis. Screen coordinates tilt the ray direction among the axes pairwise (u trades x↔y against z↔w, v trades x/z against y/w), with a `tanh` soft-clamp bounding the lens like a soft fisheye. A 2D screen can only address a 2-parameter family of the 3-sphere of possible 4D ray directions; this projection is one way to make that trade-off.

## Controls

Translations slide along a single axis of 4-space. Rotations are **planar 4D rotations** — in 4D, a rotation fixes a plane rather than an axis, and there are six independent rotation planes (one for each pair of axes).

### Movement

| Key | Direction | Key | Direction |
|---|---|---|---|
| <kbd>W</kbd> / <kbd>S</kbd> | −z / +z | <kbd>Q</kbd> / <kbd>E</kbd> | −x / +x |
| <kbd>A</kbd> / <kbd>D</kbd> | −y / +y | <kbd>Z</kbd> / <kbd>X</kbd> | −w / +w |

### Rotation (six planes)

| Keys | Plane | Keys | Plane |
|---|---|---|---|
| <kbd>R</kbd> / <kbd>T</kbd> | x–y | <kbd>V</kbd> / <kbd>B</kbd> | y–z |
| <kbd>F</kbd> / <kbd>G</kbd> | x–z | <kbd>N</kbd> / <kbd>M</kbd> | z–w |
| <kbd>↑</kbd> / <kbd>↓</kbd> | x–w | <kbd>←</kbd> / <kbd>→</kbd> | y–w |

### Other

| Key | Action |
|---|---|
| <kbd>1</kbd> | Toggle hypersphere |
| <kbd>2</kbd> | Toggle tesseract |
| <kbd>Esc</kbd> | Quit |

## Running it

Requires a working [Rust toolchain](https://rustup.rs/) and a GPU/driver with Vulkan, Metal, DX12, or OpenGL ES 3 support (wgpu picks a backend automatically).

```bash
git clone https://github.com/James-Bubear/visualise_4d.git
cd visualise_4d
cargo run --release
```

The debug build has noticeably worse frame pacing than release.

## How it works

The host side (`src/app.rs`) opens a window via winit, initializes a wgpu surface/device/pipeline, reads the keyboard into a 4D camera state, and uploads that state as a single uniform buffer every frame. The camera is a position `Vec4` plus an orientation `Mat4` built from composed Givens rotation matrices (`src/math.rs` provides a small hand-rolled 4D linear algebra library: `rot4(i, j, θ)` is a rotation in the *ij*-plane). There is no vertex data — the vertex shader emits one oversized triangle covering the screen, and all rendering happens per-pixel in `shader.wgsl`.

Each fragment computes a 4D ray direction from its screen coordinate using the symmetric diagonal lens described above, rotates it into world space by the camera basis, then sphere-traces through the scene's signed distance field: a hypersphere (radius 1.5 at the origin) and a tesseract (half-extent 1 at `[3, 0, 0, 0]`), the latter using the standard "cheap box" SDF generalized to four dimensions. The marcher takes adaptive damped steps (full distance mid-range, scaled down near and far) out to 200 steps. On a hit, a 4D surface normal is computed by central differencing of the distance field along all four axes (8 samples, one pair per axis), and the surface is shaded diffusely with a per-ray headlight — light travels along the view ray, so whatever the camera looks at is lit. A fixed world-space light instead tends to leave the camera backlit after rotations through planes involving w. An earlier CPU-only implementation lives in `src/scene.rs` as a mirror of the shader's scene logic, kept for offline testing; the GPU path is the one that renders.

## Project layout

```
├── Cargo.toml
├── shader.wgsl        # all rendering: lens, raymarch, SDFs, shading
└── src/
    ├── main.rs        # entry point, winit event loop
    ├── app.rs         # window/GPU setup, input handling, camera state
    ├── math.rs        # Vec4/Mat4, Givens rotations (rot4), vector helpers
    └── scene.rs       # CPU mirror of the shader's scene (offline/testing)
```

## Notes & known quirks

- **Two-parameter lens:** a 2D screen can only address a 2-surface of 4D directions. Rotating the camera (especially through planes involving w) changes *which* slice of 4-space you see rather than turning the view.
- **Movement is axis-aligned, not view-aligned:** translation keys slide along fixed world axes rather than relative to the camera, since a 4D camera has no unique "forward."
- The `Surface<'static>` transmute in `app.rs` stores a window-borrowing surface inside the struct that owns the window; it is sound under this program's ownership structure.
- Animation and input are paced with real delta time and clamped, so speed is framerate-independent.

## License

MIT — see [LICENSE](LICENSE).
