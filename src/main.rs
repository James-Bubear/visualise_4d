mod math;
mod scene; // We won't actually use this file for rendering, but keep it if you want CPU fallback later
mod app;

use winit::event_loop::{ControlFlow, EventLoop};
use app::App;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}