#version 460
#extension GL_EXT_ray_tracing : require

#include "hit.glsl"

layout(location = 0) rayPayloadInEXT SurfaceHit sh;

void main() {
  sh.miss = true;
}
