#version 460
#extension GL_EXT_ray_tracing : require

#include "hit.glsl"
#include "raytrace_structures.glsl"
#include "raytrace_commons.glsl"

hitAttributeEXT vec2 attrib;

layout(location = 0)  rayPayloadInEXT SurfaceHit sh;

layout(std430, set=1, binding=3) readonly buffer instanceBuffer {
  RTInstance instances[];
} InstanceBuffer;

void main()
{
  RTInstance instance = InstanceBuffer.instances[gl_InstanceCustomIndexEXT];
  uint triangle_id = instance.index_offset/3+gl_PrimitiveID;
  sh.attribs = attrib;
  sh.ids = uvec2(triangle_id, instance.material_id);
  sh.distance = gl_HitTEXT;
  sh.miss = false;
}
