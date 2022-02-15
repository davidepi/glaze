#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT bool was_hit;

void main() {
  was_hit = false;
}
