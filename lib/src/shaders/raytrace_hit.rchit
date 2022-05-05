#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier: require

#include "raytrace_structures.glsl"
#include "raytrace_commons.glsl"

hitAttributeEXT vec2 attrib;

layout(location = 0)  rayPayloadInEXT HitData hit;

layout(std430, set=1, binding=1) readonly buffer vertexBuffer {
  VertexPacked vertices[];
} VertexBuffer;
layout(std430, set=1, binding=2) readonly buffer indexBuffer {
  Triangle indices[];
} IndexBuffer;
layout(std430, set=1, binding=3) readonly buffer instanceBuffer {
  RTInstance instances[];
} InstanceBuffer;
layout(std430, set=1, binding=4) readonly buffer materialBuffer {
  RTMaterial materials[];
} MaterialBuffer;
layout(set=1, binding=6) uniform sampler2D textures[];
layout(set=1, binding=7) readonly buffer derivativeBuffer {
  Derivatives d[];
}DerivativeBuffer;


void main()
{
  RTInstance instance = InstanceBuffer.instances[gl_InstanceCustomIndexEXT];
  uint triangle_id = instance.index_offset/3+gl_PrimitiveID;
  const vec3 barycentric = vec3(1.0 - attrib.x - attrib.y, attrib.x, attrib.y);
  Triangle tris = IndexBuffer.indices[triangle_id];
  Vertex v0 = from_packed(VertexBuffer.vertices[tris.x]);
  Vertex v1 = from_packed(VertexBuffer.vertices[tris.y]);
  Vertex v2 = from_packed(VertexBuffer.vertices[tris.z]);
  hit.point = v0.position * barycentric.x +
            v1.position * barycentric.y +
            v2.position * barycentric.z;
  hit.uv = v0.tcoord * barycentric.x +
         v1.tcoord * barycentric.y +
         v2.tcoord * barycentric.z;
  hit.geometric_normal = DerivativeBuffer.d[triangle_id].normal.xyz;
  hit.dpdu = DerivativeBuffer.d[triangle_id].dpdu.xyz;
  hit.dpdv = DerivativeBuffer.d[triangle_id].dpdv.xyz;
  hit.shading_normal = v0.normal * barycentric.x +
                     v1.normal * barycentric.y +
                     v2.normal * barycentric.z;
  hit.material_id = instance.material_id;
  RTMaterial material = MaterialBuffer.materials[hit.material_id];
  if(material.normal != 0)
  {
    vec3 normal = texture(textures[nonuniformEXT(material.normal)], hit.uv).xyz;
    ShadingSpace old;
    old.s = normalize(hit.dpdu);
    old.n = hit.shading_normal;
    old.t = normalize(cross(old.n, old.s));
    hit.shading_normal = normalize(to_world_space(normal * 2.0 - 1.0, old));
    hit.shading_normal *= sign(dot(hit.geometric_normal, hit.shading_normal));
  }
  hit.distance = gl_HitTEXT;
  // convert to world space
  hit.point = gl_ObjectToWorldEXT * vec4(hit.point, 1.0);
  hit.dpdu = gl_ObjectToWorldEXT * vec4(hit.dpdu, 1.0);
  hit.dpdv = gl_ObjectToWorldEXT * vec4(hit.dpdv, 1.0);
  mat3x4 normal_obj_to_world = transpose(gl_WorldToObjectEXT);
  hit.geometric_normal = (normal_obj_to_world * hit.geometric_normal).xyz;
  hit.shading_normal = (normal_obj_to_world * hit.shading_normal).xyz;
}
