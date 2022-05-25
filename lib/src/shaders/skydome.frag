#version 460

#define ONE_OVER_PI 0.31830988618

layout(location = 0) in vec3 in_vv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding=0) uniform sampler2D diffuse;

void main() {
    float u = 0.5 - 0.5 * atan(in_vv.x, -in_vv.z) * ONE_OVER_PI;
    float v = 1.0 - acos(in_vv.y) * ONE_OVER_PI;
    vec3 diff = texture(diffuse, vec2(u, v)).xyz;
    out_color = vec4(diff, 1.0);
}
