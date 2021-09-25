use shaderc;
use shaderc::{CompileOptions, Compiler, OptimizationLevel, ShaderKind};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=src/shaders");
    let is_debug = cfg!(debug_assertions);
    let config = if is_debug { "debug" } else { "release" };
    // target MUST be hardcoded, otherwise I cannot use include! macro
    let outdir = PathBuf::from("target").join(config).join("shaders");
    std::fs::create_dir_all(outdir.clone())?;
    let mut compiler = Compiler::new().expect("Failed to find a SPIR-V compiler");
    let mut options = CompileOptions::new().expect("Error while initializing compiler");
    if is_debug {
        options.set_optimization_level(OptimizationLevel::Zero);
    } else {
        options.set_optimization_level(OptimizationLevel::Performance);
    }
    for entry in std::fs::read_dir("src/shaders")? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let in_path = entry.path();
            let maybe_correct_kind =
                in_path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(ShaderKind::Vertex),
                        "comp" => Some(ShaderKind::Compute),
                        "frag" => Some(ShaderKind::Fragment),
                        _ => None,
                    });
            if let Some(shader_kind) = maybe_correct_kind {
                let source_text = std::fs::read_to_string(&in_path)?;
                let compiled_bytes = compiler.compile_into_spirv(
                    &source_text,
                    shader_kind,
                    in_path.as_path().to_str().unwrap(),
                    "main",
                    Some(&options),
                )?;
                let outfile = outdir.clone().join(format!(
                    "{}.spv",
                    in_path.as_path().file_name().unwrap().to_str().unwrap()
                ));

                std::fs::write(&outfile, &compiled_bytes.as_binary_u8())?;
            }
        }
    }
    Ok(())
}
