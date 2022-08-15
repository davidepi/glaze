#![cfg_attr(
    all(target_os = "windows", not(feature = "console"),),
    windows_subsystem = "windows"
)]
use clap::Parser;
use viewport::InteractiveView;
use winit::event_loop::EventLoop;

mod ui;
mod viewport;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The scene that will be loaded on startup.
    scene: Option<String>,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let event_loop = EventLoop::new();
    let iw = InteractiveView::new(&event_loop, args.scene);
    match iw {
        Ok(iw) => {
            iw.main_loop(event_loop);
        }
        Err(e) => {
            log::error!("{}", e);
        }
    }
}
