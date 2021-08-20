use glaze::GlazeApp;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let glaze = GlazeApp::new(&event_loop);
    glaze.main_loop(event_loop);
}