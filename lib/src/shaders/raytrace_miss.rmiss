#version 460
#extension GL_EXT_ray_tracing : require

#include "raytrace_commons.glsl"

layout(location = 0) rayPayloadInEXT HitPoint hit;

void main() {
  hit.miss = true;
}
