#ifndef _COMMONS_GLSL_
#define _COMMONS_GLSL_

#include "spectrum.glsl"
#include "shading_space.glsl"


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

struct Triangle {
  uint x;
  uint y;
  uint z;
};

Vertex from_packed(VertexPacked vp) {
  vec3 normal = vec3(vp.pxyznx.w, vp.nyztxy.xy);
  return Vertex(vp.pxyznx.xyz, normal, vp.nyztxy.zw);
}

struct Derivatives
{
  vec4 normal; // used in SSBO so vec4 :(
  vec4 dpdu;
  vec4 dpdv;
};

struct HitPoint
{
  vec3 point;
  vec3 shading_normal;
  vec3 geometric_normal;
  vec3 dpdu;
  vec3 dpdv;
  vec2 uv;
  uint material_index;
  float distance;
  bool miss;
};

struct SampledLight
{
  Spectrum emission;
  vec3 position;
  float pdf;
  vec3 wiW;
  float distance;
  uint light_index;
};

struct RTLight
{
  vec4 col0;
  vec4 col1;
  vec4 col2;
  vec4 col3;
  vec4 pos;
  vec4 dir;
  uint shader;
};

struct RTMaterial
{
  vec4 diffuse_mul;
  uint diffuse;
  uint opacity;
  uint normal;
  uint bsdf_index;
  vec4 ior0;
  vec4 ior1;
  vec4 ior2;
  vec4 ior3;
  vec4 metal_fresnel0;
  vec4 metal_fresnel1;
  vec4 metal_fresnel2;
  vec4 metal_fresnel3;
  bool is_specular;
};

struct BsdfValue
{
  vec3 woW;
  vec3 wiW;
  vec3 geometric_normal;
  vec2 uv;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
};

struct BsdfSampleValue
{
  vec3 woW;
  vec3 wiW;
  vec2 uv;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
  vec2 rand_sample;
  float pdf;
  bool was_spec;
};

struct BsdfPdf
{
  vec3 woW;
  vec3 wiW;
  ShadingSpace shading;
  float pdf;
};

#endif
