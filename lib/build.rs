#[cfg(feature = "vulkan")]
use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeCallbackResult, IncludeType, OptimizationLevel,
    ResolvedInclude, ShaderKind, TargetEnv,
};
#[cfg(feature = "vulkan")]
use std::collections::HashMap;
#[cfg(feature = "vulkan")]
use std::env;
use std::error::Error;
#[cfg(feature = "vulkan")]
use std::fmt::Write;
#[cfg(feature = "vulkan")]
use std::path::Path;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;
#[cfg(feature = "vulkan")]
use syn::{Expr, Item, ItemConst, Lit, Type, TypeArray, TypePath};

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "vulkan")]
    {
        println!("cargo:rerun-if-changed=src/vulkan/raytrace_structures.rs");
        println!("cargo:rerun-if-changed=src/shaders");
        gen_shared_structures(&["src/vulkan/raytrace_structures.rs"])?;
        let variants = [
            ("path_trace.rgen", "", "path_trace.rgen"),
            ("path_trace.rgen", "DIRECT_ONLY", "direct.rgen"),
        ];
        compile_spirv(&variants)?;
    }
    Ok(())
}

// variants is an array of tuples (input_file_name, define, output_file_name)
// if a file matches the first element in the tuple, the second element will be the list of defines
// and the third the output filename. This will replace the default "empty defines" variant.
#[cfg(feature = "vulkan")]
fn compile_spirv(variants: &[(&str, &str, &str)]) -> Result<(), Box<dyn Error>> {
    let is_debug = cfg!(debug_assertions);
    // specifying different folders is not necessary with OUT_DIR, cargo handles this automatically
    let outdir = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("shaders");
    let mut variants_map = HashMap::new();
    for (inname, define, outname) in variants {
        variants_map
            .entry(inname)
            .or_insert_with(Vec::new)
            .push((*define, *outname));
    }
    std::fs::create_dir_all(outdir.clone())?;
    let compiler = Compiler::new().expect("Failed to find a SPIR-V compiler");
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
                let in_name = in_path.file_name().unwrap().to_str().unwrap();
                let default_define = vec![("", in_name)];
                let defines = variants_map.get(&in_name).unwrap_or(&default_define);
                for (define, out_name) in defines {
                    let source_text = std::fs::read_to_string(&in_path)?;
                    // thank you google for using the same name of a method in the prelude (std::clone)
                    // but a different signature.
                    // much appreciated
                    let mut edited_options = options.clone().unwrap();
                    if !define.is_empty() {
                        edited_options.add_macro_definition(define, None);
                    }
                    let compiled_bytes = compiler.compile_into_spirv(
                        &source_text,
                        shader_kind,
                        in_path.as_path().to_str().unwrap(),
                        "main",
                        Some(&edited_options),
                    )?;
                    let outfile = outdir.clone().join(format!("{}.spv", out_name));
                    std::fs::write(&outfile, &compiled_bytes.as_binary_u8())?;
                }
            }
        }
    }
    Ok(())
}

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

#[cfg(feature = "vulkan")]
fn gen_shared_structures(files: &[&str]) -> Result<(), Box<dyn Error>> {
    for file in files {
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)?;
        let filestem = Path::new(file).file_stem().unwrap().to_str().unwrap();
        let output = format!("src/shaders/{}.glsl", filestem);
        let guard_name = format!("_{}_GLSL_", filestem.to_uppercase());
        let mut output_content = format!("// Automatically generated by build.rs from {}\n", file);
        output_content.push_str("// This file will be automatically re-generated by cargo\n\n");
        writeln!(&mut output_content, "#ifndef {guard_name}")?;
        writeln!(&mut output_content, "#define {guard_name}")?;
        output_content.push_str("#include \"spectrum.glsl\"\n");
        let mut structs = Vec::new();
        let mut consts = Vec::new();
        ast.items.into_iter().for_each(|i| match i {
            Item::Struct(s) => structs.push(s),
            Item::Const(c) => consts.push(c),
            _ => (),
        });
        output_content.push('\n');
        for con in consts {
            let define = const_to_c_define(con);
            output_content.push_str(&define);
        }
        output_content.push('\n');
        for stru in structs {
            writeln!(&mut output_content, "struct {}", stru.ident)?;
            output_content.push_str("{\n");
            for field in stru.fields {
                let field_name = field
                    .ident
                    .expect("Unnamed fields are not supported in GLSL")
                    .to_string();
                let field_c = match field.ty {
                    Type::Path(tp) => path_type_to_c_type(tp, field_name),
                    Type::Array(tp) => array_type_to_c_type(tp, field_name),
                    _ => panic!("Type not supported in GLSL"),
                };
                output_content.push_str(&field_c);
            }
            output_content.push_str("};\n\n");
        }
        output_content.push_str("#endif\n\n");
        std::fs::write(output, output_content)?;
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn path_type_to_c_type(path: TypePath, field_name: String) -> String {
    let last = path.path.segments.last().unwrap();
    let name = last.ident.to_string();
    let ty = match name.as_str() {
        "u32" => "uint",
        "f32" => "float",
        "bool" => "bool",
        "Vector2" | "Point2" => "vec2", // not checking that this is really f32
        "Vector3" | "Point3" => "vec3", // I will face a build error if the type is wrong
        "Vector4" | "Point4" => "vec4", // and will update this file in that case
        "Matrix4" => "mat4",
        "Spectrum" => "Spectrum",
        _ => panic!(
            "Type {} is not supported in GLSL (or in the build.rs script)",
            name
        ),
    };
    format!("  {ty} {field_name};\n")
}

#[cfg(feature = "vulkan")]
fn array_type_to_c_type(array: TypeArray, field_name: String) -> String {
    let inner_type = match *array.elem {
        Type::Path(tp) => tp,
        _ => panic!("Nested arrays are not supported in GLSL"),
    };
    let ty = inner_type.path.segments.last().unwrap().ident.to_string();
    let len = match array.len {
        Expr::Lit(l) => match l.lit {
            Lit::Int(l) => l.to_string(),
            _ => panic!("Array lenght must be an integer"),
        },
        Expr::Path(p) => p
            .path
            .get_ident()
            .expect("Failed to get array length")
            .to_string(),
        _ => panic!("Unsupported array length"),
    };
    let len_number = len.parse::<usize>().unwrap_or(usize::MAX);
    if len_number <= 4 && (ty == "u32" || ty == "f32") {
        let prefix = if ty == "u32" { "u" } else { "" };
        match len_number {
            0 => panic!("Zero length array are not supported in GLSL"),
            1 => panic!("Use a primitive type, not a single valued array"),
            2 => format!("  {prefix}vec2 {field_name};\n"),
            3 => format!("  {prefix}vec3 {field_name};\n"),
            4 => format!("  {prefix}vec4 {field_name};\n"),
            _ => panic!(), // unreachable
        }
    } else {
        match ty.as_str() {
            "u32" => format!("  uint {field_name}[{len}];\n"),
            "f32" => format!("  float {field_name}[{len}];\n"),
            _ => format!("  {ty} {field_name}[{len}];\n"),
        }
    }
}

#[cfg(feature = "vulkan")]
fn const_to_c_define(con: ItemConst) -> String {
    match *con.ty {
        Type::Path(_) => {
            let val = match *con.expr {
                Expr::Lit(l) => match l.lit {
                    Lit::Int(l) => l.to_string(),
                    Lit::Float(f) => f.to_string(),
                    _ => panic!("Constant value must be an integer or a float"),
                },
                _ => panic!("Constant value must be a literal"),
            };
            format!("#define {} {}\n", con.ident, val)
        }
        _ => panic!("Only literals are supported as conts"),
    }
}
