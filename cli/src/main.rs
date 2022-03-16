use clap::Parser;
use console::style;
use glaze::{parse, RayTraceInstance, RayTraceRenderer, RayTraceScene};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input scene to be rendered.
    input: String,
    /// Output image (.jpg or .png only).
    output: String,
    /// Rendering resolution in form "WxH".
    #[clap(short, long = "res", default_value = "1920x1080")]
    resolution: String,
    /// Samples per pixel.
    #[clap(short, long, default_value = "256")]
    spp: usize,
}

fn main() {
    let args = Args::parse();
    env_logger::init();
    let width;
    let height;
    // check output
    if !(args.output.ends_with("jpg") || args.output.ends_with("png")) {
        log::error!("The output image must end with .jpg or .png");
        std::process::exit(1);
    }
    let out_path = Path::new(&args.output);
    if std::fs::File::create(out_path).is_err() {
        log::error!("The output file can not be written");
        std::process::exit(1);
    }
    // check resolution
    if let Some(pos) = args.resolution.find('x') {
        let wstr = &args.resolution[..pos];
        let hstr = &args.resolution[pos + 1..];
        if let Ok(parsedw) = wstr.parse() {
            width = parsedw;
        } else {
            log::error!("Failed to parse the requested width");
            std::process::exit(1);
        }
        if let Ok(parsedh) = hstr.parse() {
            height = parsedh;
        } else {
            log::error!("Failed to parse the requested height");
            std::process::exit(1);
        }
    } else {
        log::error!("The resolution must be specified in form WIDTHxHEIGHT, for example 1920x1080");
        std::process::exit(1);
    }
    if let Some(instance) = RayTraceInstance::new() {
        let path = PathBuf::from(&args.input);
        match parse(path) {
            Ok(parsed) => {
                let pb = ProgressBar::new_spinner();
                let sty = ProgressStyle::default_spinner()
                    .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈")
                    .template("{msg} {spinner}")
                    .unwrap();
                pb.set_style(sty);
                let stage_msg = "Parsing and setting up scene... ";
                pb.set_message(stage_msg);
                pb.enable_steady_tick(Duration::from_millis(120));
                let setup_start = Instant::now();
                let instance = Arc::new(instance);
                let scene = RayTraceScene::<RayTraceInstance>::new(Arc::clone(&instance), parsed);
                let mut renderer =
                    RayTraceRenderer::<RayTraceInstance>::new(instance, Some(scene), width, height);
                let setup_end = Instant::now();
                pb.finish_with_message(format!(
                    "{}{} ({} ms)",
                    stage_msg,
                    style("Done").bold().green(),
                    (setup_end - setup_start).as_millis()
                ));
                let sty = ProgressStyle::default_bar()
                    .progress_chars("#>-")
                    .template("{msg} {pos:>7}/{len:7} {bar:40.cyan/blue} [{elapsed_precise}]")
                    .unwrap();
                let stage_msg = format!("Rendering @ {}x{}... ", width, height);
                let pb = ProgressBar::new(args.spp as u64);
                pb.set_style(sty);
                pb.set_message(stage_msg.clone());
                let render_start = Instant::now();
                let image = renderer.draw(
                    args.spp,
                    Some(|| {
                        pb.inc(1);
                    }),
                );
                let render_end = Instant::now();
                pb.finish_with_message(format!(
                    "{}{} ({} ms)",
                    stage_msg,
                    style("Done").bold().green(),
                    (render_end - render_start).as_millis()
                ));
                if let Err(e) = image.save(args.output) {
                    log::error!("Failed to save image: {}", e);
                } else {
                    println!("All done :)");
                }
            }
            Err(io) => {
                log::error!("{io}");
                std::process::exit(1);
            }
        }
    } else {
        log::error!("Cannot create the Vulkan instance. Does your GPU support raytracing?");
    }
}
