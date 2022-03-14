#ifndef _COMMONS_GLSL_
#define _COMMONS_GLSL_

#include "spectrum.glsl"
#include "shading_space.glsl"

#define CHECKNAN(X) (isnan(X)?0.0:X)
#define CHECKINF(X) (isinf(X)?0.0:X)
#define CHECKNANINF(X) (isnan(X)||isinf(X)?0.0:X)

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
  vec4 ior0;
  vec4 ior1;
  vec4 ior2;
  vec4 ior3;
  vec4 metal_fresnel0;
  vec4 metal_fresnel1;
  vec4 metal_fresnel2;
  vec4 metal_fresnel3;
  uint diffuse;
  uint roughness;
  uint metalness;
  uint opacity;
  uint normal;
  uint bsdf_index;
  float roughness_mul;
  float metalness_mul;
  float anisotropy;
  float ior_dielectric;
  bool is_specular;
};

struct BsdfValue
{
  vec3 woW;
  vec3 wiW;
  vec3 geometric_normal;
  vec2 uv;
  float rand_sample;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
  float pdf;
};

struct BsdfSampleValue
{
  vec3 woW;
  vec3 wiW;
  vec3 geometric_normal;
  vec2 uv;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
  vec3 rand_sample;
  float pdf;
};

#endif
