use viewport::InteractiveView;
use winit::event_loop::EventLoop;

mod ui;
mod viewport;

fn main() {
    env_logger::init();
    let mut event_loop = EventLoop::new();
    let iw = InteractiveView::new(&mut event_loop);
    match iw {
        Ok(mut iw) => {
            iw.main_loop(&mut event_loop);
            iw.destroy();
        }
        Err(e) => {
            log::error!("{}", e);
        }
    }
}
