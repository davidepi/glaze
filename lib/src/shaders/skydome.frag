#version 460

layout(location = 0) in vec2 in_vt;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding=0) uniform sampler2D diffuse;

void main() {
    vec3 diff = texture(diffuse, in_vt).xyz;
    out_color = vec4(diff, 1.0);
}
