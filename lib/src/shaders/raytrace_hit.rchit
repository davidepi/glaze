#version 460
#extension GL_EXT_ray_tracing : require

#include "raytrace_commons.glsl"

hitAttributeEXT vec2 attrib;

layout(location = 0)  rayPayloadInEXT HitPoint hit;

layout(std430, set=1, binding=1) readonly buffer vertexBuffer {
  VertexPacked vertices[];
} VertexBuffer;

layout(std430, set=1, binding=2) readonly buffer indexBuffer {
  Triangle indices[];
} IndexBuffer;

layout(std430, set=1, binding=3) readonly buffer instanceBuffer {
  RTInstance instances[];
} InstanceBuffer;

layout(std430, set=1, binding=7) readonly buffer derivativeBuffer {
  Derivatives d[];
} DerivativeBuffer;

void main()
{
  const vec3 barycentric = vec3(1.0 - attrib.x - attrib.y, attrib.x, attrib.y);
  RTInstance instance = InstanceBuffer.instances[gl_InstanceCustomIndexEXT];
  uint triangle_id = instance.index_offset/3+gl_PrimitiveID;
  Triangle tris = IndexBuffer.indices[triangle_id];
  Vertex v0 = from_packed(VertexBuffer.vertices[tris.x]);
  Vertex v1 = from_packed(VertexBuffer.vertices[tris.y]);
  Vertex v2 = from_packed(VertexBuffer.vertices[tris.z]);

  hit.point = v0.position * barycentric.x + v1.position * barycentric.y + v2.position * barycentric.z;
  hit.shading_normal = v0.normal * barycentric.x + v1.normal * barycentric.y + v2.normal * barycentric.z;
  hit.uv = v0.tcoord * barycentric.x + v1.tcoord * barycentric.y + v2.tcoord * barycentric.z;
  hit.distance = gl_HitTEXT;
  hit.material_index = instance.mat_id;
  hit.geometric_normal = DerivativeBuffer.d[triangle_id].normal.xyz;
  hit.dpdu = DerivativeBuffer.d[triangle_id].dpdu.xyz;
  hit.dpdv = DerivativeBuffer.d[triangle_id].dpdv.xyz;
  hit.miss = false;
}
