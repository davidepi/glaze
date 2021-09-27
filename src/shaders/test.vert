#version 460

layout(location = 0) in vec3 in_vv;
layout(location = 1) in vec3 in_vn;
layout(location = 2) in vec2 in_vt;

layout(location = 0) out vec3 out_vn;
layout(location = 1) out vec2 out_vt;

void main() {
    gl_Position = vec4(in_vv, 1.0);
    out_vn = in_vn;
    out_vt = in_vt;
}