#version 460

layout(location = 0) in vec3 in_vn;
layout(location = 1) in vec2 in_vt;

layout(location = 0) out vec4 out_color;

layout(set=0, binding=0) uniform fd {
    mat4 projview;
    float frame_time;
} FrameData;
layout(set = 1, binding=0) uniform mp {
    vec3 diffuse_color;
} MaterialParams;
layout(set = 1, binding=1) uniform sampler2D diffuse;
layout(set = 1, binding=2) uniform sampler2D opacity;

void main() {
    if (texture(opacity, in_vt).x > 0.5) 
    {
        vec3 diff = texture(diffuse, in_vt).xyz * MaterialParams.diffuse_color.xyz;
        out_color = vec4(diff, 1.0);
    } 
    else 
    {
        discard; // not the best approach, but for this simple viewport is enough
        // maybe in the future I will implement proper transparency, for now, this is enough
    }
}
