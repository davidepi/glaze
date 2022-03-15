use clap::{App, Arg};
use console::style;
use faccess::PathExt;
use glaze::{parse, RayTraceInstance, RayTraceRenderer, RayTraceScene};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Instant;

fn main() {
    env_logger::init();
    let matches = App::new("glaze-cli")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about("GPU-based 3D renderer")
        .arg(
            Arg::new("input")
                .required(true)
                .help("The input file. Must have been generated with glaze-converter."),
        )
        .arg(
            Arg::new("output")
                .required(true)
                .help("The output file. Only .png or .jpg are supported."),
        )
        .arg(
            Arg::new("resolution")
                .short('r')
                .long("res")
                .help("The output resolution in form WxH.")
                .default_value("1920x1080"),
        )
        .get_matches();

    let input = matches.value_of("input").unwrap();
    let output = matches.value_of("output").unwrap();
    let res = matches.value_of("resolution").unwrap();
    let width;
    let height;
    // check output
    if !(output.ends_with("jpg") || output.ends_with("png")) {
        log::error!("The output image must end with .jpg or .png");
        std::process::exit(1);
    }
    let out_path = Path::new(output);
    if !out_path.writable() {
        log::error!("The output file can not be written");
        std::process::exit(1);
    }
    // check resolution
    if let Some(pos) = res.find('x') {
        let wstr = &res[..pos];
        let hstr = &res[pos + 1..];
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
        let path = PathBuf::from(input);
        match parse(path) {
            Ok(parsed) => {
                print!("Parsing and setting up scene... ");
                let setup_start = Instant::now();
                let instance = Arc::new(instance);
                let scene = RayTraceScene::<RayTraceInstance>::new(Arc::clone(&instance), parsed);
                let renderer =
                    RayTraceRenderer::<RayTraceInstance>::new(instance, Some(scene), width, height);
                let setup_end = Instant::now();
                println!(
                    "{} ({} ms)",
                    style("Done").bold().green(),
                    (setup_end - setup_start).as_millis()
                );
                let (write, _read) = mpsc::channel();
                print!("Rendering @ {}x{}... ", width, height);
                let render_start = Instant::now();
                let image = renderer.draw(write).export();
                let render_end = Instant::now();
                println!(
                    "{} ({} ms)",
                    style("Done").bold().green(),
                    (render_end - render_start).as_millis()
                );
                if let Err(e) = image.save(output) {
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
