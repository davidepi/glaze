# Glaze
![build badge](https://github.com/davidepi/glaze/actions/workflows/ci.yml/badge.svg)

Realtime path tracing renderer written in Rust and Vulkan Ray Tracing.

Requires a GPU with raytracing support.

![test_image](https://user-images.githubusercontent.com/2979535/204575102-142d4e3f-365b-4e15-b754-e21fec08eccb.jpg)

The following is a small video showing the realtime capabilities of the renderer.

**Note:** flickering is expected during the first few frames, given the unbiased nature of Path Tracing.

https://user-images.githubusercontent.com/2979535/204574735-59aff2a5-234d-4ea5-a167-dc108c7e9543.mp4

### Supported Materials

| Material type | Preview                                                      |
|---------------|--------------------------------------------------------------|
| Lambert       | <img src="resources/readme/materials/lambert.jpg" width=128/>|
| Glass         | <img src="resources/readme/materials/glass.jpg" width=128/>  |
| Mirror        | <img src="resources/readme/materials/mirror.jpg" width=128/> |
| Metal         | <img src="resources/readme/materials/metal.jpg" width=128/>  |
| Pbr (GGX)     | <img src="resources/readme/materials/pbr.jpg" width=128/>    |

## Building and Running
### Building
_Glaze_ works on Windows and Linux operating systems.

In order to build this application
[Rust](https://www.rust-lang.org/tools/install) is required.

The following command can be used to build the interactive renderer:
```bash
cargo build --release --bin glaze-app
```
The executable can be found in the folder `target/release`.

The only runtime dependency is a decently recent version of the Vulkan runtime,
usually bundled with the video card graphics driver.


To build the non-interactive renderer or the 3D scene converter, `glaze-app` in
the previous command should be replaced with `glaze-cli` or `glaze-converter`
respectively. Note that building the converter requires
[assimp](https://github.com/assimp/assimp.git) to be installed in the system.
For non-interactive executables (`glaze-cli` and `glaze-converter`),
command line parameters can be retrieved with the `-h` flags.

### Input file
This renderer requires its own 3D format.
(Support for [glTF](https://github.com/KhronosGroup/glTF) SOON™)

A (very) experimental converter, `glaze-converter`, based on
[assimp](https://github.com/assimp), can be used to convert a 3D model from other
formats to the one expected by this renderer.

### Example scene

The scene used in this README can be downloaded at the
[following address](https://sel.ist.osaka-u.ac.jp/people/davidepi/sponza.glaze).

## Repo Structure
- *[lib](lib)*: library containing all the rendering and parsing routines.
- *[converter](converter)*: executable responsible of converting existing 3D model files into the
custom format required by this project.
- *[app](app)*: executable providing an interactive application to view the rendered scene and modify lights and materials in real-time.
- [cli](cli): executable used to render a scene non-interactively. Still requires a GPU with Ray Tracing support.
