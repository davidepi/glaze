#version 460

layout (set = 0,binding = 0) uniform sampler2D input_texture;

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec4 out_color;

void main() 
{
	out_color = texture(input_texture, in_uv);	
}