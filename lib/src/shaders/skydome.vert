#version 460

layout(location = 0) in vec3 in_vv;
layout(location = 1) in vec2 in_vt;

layout(location = 0) out vec2 out_vt;

layout(push_constant) uniform readonly pc {
  mat4 mvp;
} PushConstant;

void main() {
    gl_Position = PushConstant.mvp * vec4(in_vv, 1.0);
    out_vt = in_vt;
}
