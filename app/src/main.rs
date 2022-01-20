use viewport::InteractiveView;
use winit::event_loop::EventLoop;

mod ui;
mod viewport;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let iw = InteractiveView::new(&event_loop);
    match iw {
        Ok(iw) => {
            iw.main_loop(event_loop);
        }
        Err(e) => {
            log::error!("{}", e);
        }
    }
}
