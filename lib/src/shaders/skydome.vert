#version 460

layout(location = 0) in vec3 in_vv;

layout(location = 0) out vec3 out_vv;

layout(push_constant) uniform readonly pc {
  mat4 mvp;
} PushConstant;

void main() {
    gl_Position = PushConstant.mvp * vec4(in_vv, 1.0);
    out_vv = in_vv;
}
