#version 460

#include "constants.glsl"

layout(location = 0) in vec3 in_vv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding=0) uniform sampler2D diffuse;

void main() {
  float phi = atan(in_vv.y, in_vv.x); // just call it atan2 like everybody else...
  float theta = acos(in_vv.z);
  vec2 uv = vec2(phi * INV_2PI, theta * INV_PI);
  vec3 diff = texture(diffuse, uv).xyz;
  out_color = vec4(diff, 1.0);
}
