#version 460

layout(location = 0) in vec3 in_vn;
layout(location = 1) in vec2 in_vt;

layout(location = 0) out vec4 out_color;

layout(set=0, binding=0) uniform fd {
    mat4 projview;
    float frame_time;
} FrameData;

void main() {
    out_color = vec4(abs(sin(FrameData.frame_time)),abs(cos(FrameData.frame_time)),0.0,0.0);
}