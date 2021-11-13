#version 460

layout(location = 0) in vec3 in_vn;
layout(location = 1) in vec2 in_vt;

layout(location = 0) out vec4 out_color;

layout(set=0, binding=0) uniform fd {
    mat4 projview;
    float frame_time;
} FrameData;
layout(set = 1, binding=0) uniform sampler2D diffuse;

void main() {
    vec3 diff = texture(diffuse, in_vt).xyz;
    out_color = vec4(abs(sin(FrameData.frame_time))*diff.x,abs(cos(FrameData.frame_time))*diff.y,abs(sin(-FrameData.frame_time))*diff.z,1.0);
}