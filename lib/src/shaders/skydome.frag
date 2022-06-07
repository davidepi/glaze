#version 460

#include "constants.glsl"

layout(location = 0) in vec3 in_vv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding=0) uniform sampler2D diffuse;

void main() {
  float u = atan(in_vv.y, in_vv.x); // just call it atan2 like everybody else...
  u = u < 0.0 ? u + TWO_PI : u;
  vec2 uv = vec2(u, acos(in_vv.z)) * INV_2PI;
  vec3 diff = texture(diffuse, uv).xyz;
  out_color = vec4(diff, 1.0);
}
