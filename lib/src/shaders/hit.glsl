#ifndef _HIT_GLSL_
#define _HIT_GLSL_

struct HitData
{
  vec3 point;
  vec3 shading_normal;
  vec3 geometric_normal;
  vec3 dpdu;
  vec3 dpdv;
  vec2 uv;
};

struct SurfaceHit {
  vec2 attribs;
  uvec2 ids; //triangle_id, material_id
  float distance;
  bool miss;
};

// transform a SurfaceHit into a HitPoint
#define EXPAND_HIT(X, HIT, MATERIAL)                                            \
{                                                                               \
  const vec3 barycentric = vec3(                                                \
      1.0 - X.attribs.x - X.attribs.y,                                            \
      X.attribs.x,                                                               \
      X.attribs.y                                                                \
    );                                                                          \
  uint triangle_id = X.ids.x;                                                   \
  uint material_id = X.ids.y;                                                   \
  Triangle tris = IndexBuffer.indices[triangle_id];                             \
  Vertex v0 = from_packed(VertexBuffer.vertices[tris.x]);                       \
  Vertex v1 = from_packed(VertexBuffer.vertices[tris.y]);                       \
  Vertex v2 = from_packed(VertexBuffer.vertices[tris.z]);                       \
  HIT.point = v0.position * barycentric.x +                                     \
            v1.position * barycentric.y +                                       \
            v2.position * barycentric.z;                                        \
  HIT.uv = v0.tcoord * barycentric.x +                                          \
         v1.tcoord * barycentric.y +                                            \
         v2.tcoord * barycentric.z;                                             \
  HIT.geometric_normal = DerivativeBuffer.d[triangle_id].normal.xyz;            \
  HIT.dpdu = DerivativeBuffer.d[triangle_id].dpdu.xyz;                          \
  HIT.dpdv = DerivativeBuffer.d[triangle_id].dpdv.xyz;                          \
  HIT.shading_normal = v0.normal * barycentric.x +                              \
                     v1.normal * barycentric.y +                                \
                     v2.normal * barycentric.z;                                 \
  MATERIAL = MaterialBuffer.materials[material_id];                             \
  if(MATERIAL.normal != 0)                                                      \
  {                                                                             \
    vec3 normal = texture(textures[nonuniformEXT(MATERIAL.normal)], HIT.uv).xyz;\
    ShadingSpace old;                                                           \
    old.s = normalize(HIT.dpdu);                                                \
    old.n = HIT.shading_normal;                                                 \
    old.t = normalize(cross(old.n, old.s));                                     \
    HIT.shading_normal = normalize(to_world_space(normal * 2.0 - 1.0, old));    \
    HIT.shading_normal *= sign(dot(HIT.geometric_normal, HIT.shading_normal));  \
  }                                                                             \
}

#endif
