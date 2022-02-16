#version 460
#extension GL_EXT_ray_tracing : require

#include "raytrace_commons.glsl"

struct RTInstance {
  uint index_offset;
  uint index_count;
  uint mat_id;
};

struct VertexPacked {
  vec4 pxyznx; //position xyz + normal x
  vec4 nyztxy; //normal yz + tcoord xy
};

struct Vertex {
  vec3 position;
  vec3 normal;
  vec2 tcoord;
};

struct Index {
  uint x;
  uint y;
  uint z;
};

hitAttributeEXT vec2 attrib;

layout(location = 0)  rayPayloadInEXT HitPoint hit;

layout(std430, set=1, binding=1) readonly buffer vertexBuffer {
  VertexPacked vertices[];
} VertexBuffer;

layout(std430, set=1, binding=2) readonly buffer indexBuffer {
  Index indices[];
} IndexBuffer;

layout(std430, set=1, binding=3) readonly buffer instanceBuffer {
  RTInstance instances[];
} InstanceBuffer;

Vertex from_packed(VertexPacked vp) {
  vec3 normal = vec3(vp.pxyznx.w, vp.nyztxy.xy);
  return Vertex(vp.pxyznx.xyz, normal, vp.nyztxy.zw);
}

void main()
{
  const vec3 barycentric = vec3(1.0 - attrib.x - attrib.y, attrib.x, attrib.y);
  RTInstance instance = InstanceBuffer.instances[gl_InstanceCustomIndexEXT];
  Index idx = IndexBuffer.indices[instance.index_offset/3+gl_PrimitiveID];
  Vertex v0 = from_packed(VertexBuffer.vertices[idx.x]);
  Vertex v1 = from_packed(VertexBuffer.vertices[idx.y]);
  Vertex v2 = from_packed(VertexBuffer.vertices[idx.z]);

  hit.point = v0.position * barycentric.x + v1.position * barycentric.y + v2.position * barycentric.z;
  hit.normal = v0.normal * barycentric.x + v1.normal * barycentric.y + v2.normal * barycentric.z;
  hit.uv = v0.tcoord * barycentric.x + v1.tcoord * barycentric.y + v2.tcoord * barycentric.z;
  hit.distance = gl_HitTEXT;
  hit.material_index = instance.mat_id;
  hit.dpdu = v0.position - v2.position;
  hit.dpdv = v1.position - v2.position;
  hit.miss = false;
}
