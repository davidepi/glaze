#version 460

layout(location = 0) in vec3 in_vv;
layout(location = 1) in vec3 in_vn;
layout(location = 2) in vec2 in_vt;

layout(location = 0) out vec3 out_vn;
layout(location = 1) out vec2 out_vt;

layout(set=0, binding=0) uniform fd {
    mat4 projview;
    float frame_time;
} FrameData;

layout(set=2, binding=0) uniform od {
  mat4 model;
} ObjectData;

void main() {
    mat4 mvp = FrameData.projview * ObjectData.model;
    gl_Position = mvp * vec4(in_vv, 1.0);
    out_vn = in_vn;
    out_vt = in_vt;
}
