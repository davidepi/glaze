#version 460
layout(location = 0) out vec4 col;
layout(set=0, binding=0) uniform sampler2D fonts;
layout(location = 0) in struct { vec4 col; vec2 uv; } In;

void main()
{
    // used for single channel textures, to avoid the color red
    col = In.col * vec4(texture(fonts, In.uv.st).rrr, 1.0);
}