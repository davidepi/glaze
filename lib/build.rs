#[cfg(feature = "vulkan")]
use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeCallbackResult, IncludeType, OptimizationLevel,
    ResolvedInclude, ShaderKind, TargetEnv,
};
#[cfg(feature = "vulkan")]
use std::env;
use std::error::Error;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
fn handle_includes(name: &str, _: IncludeType, _: &str, _: usize) -> IncludeCallbackResult {
    let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shaders")
        .join(name);
    match file.canonicalize() {
        Ok(absolute) => match std::fs::read_to_string(absolute.clone()) {
            Ok(content) => {
                let resolved_name = absolute.to_str().unwrap().to_string();
                Ok(ResolvedInclude {
                    resolved_name,
                    content,
                })
            }
            Err(error) => Err(error.to_string()),
        },
        Err(error) => Err(error.to_string()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "vulkan")]
    {
        println!("cargo:rerun-if-changed=src/shaders");

        let is_debug = cfg!(debug_assertions);
        // specyfing different folders is not necessary with OUT_DIR, cargo handles this automatically
        let outdir = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("shaders");
        std::fs::create_dir_all(outdir.clone())?;
        let mut compiler = Compiler::new().expect("Failed to find a SPIR-V compiler");
        let mut options = CompileOptions::new().expect("Error while initializing compiler");
        options.set_include_callback(handle_includes);
        options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
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
                            "rgen" => Some(ShaderKind::RayGeneration),
                            "rahit" => Some(ShaderKind::AnyHit),
                            "rchit" => Some(ShaderKind::ClosestHit),
                            "rmiss" => Some(ShaderKind::Miss),
                            "rcall" => Some(ShaderKind::Callable),
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
    }
    Ok(())
}
