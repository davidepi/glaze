#version 460
#extension GL_EXT_ray_tracing : require

#include "constants.glsl"
#include "raytrace_commons.glsl"

layout(location = 0) rayPayloadInEXT HitData hit;

void main() {
  hit.distance = INFINITY;
}
