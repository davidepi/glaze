#ifndef _COMMONS_GLSL_
#define _COMMONS_GLSL_

#include "spectrum.glsl"
#include "shading_space.glsl"

#define CHECKNAN(X) (isnan(X)?0.0:X)
#define CHECKINF(X) (isinf(X)?0.0:X)
#define CHECKNANINF(X) (isnan(X)||isinf(X)?0.0:X)

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

struct HitData
{
  vec3 point;
  vec3 shading_normal;
  vec3 geometric_normal;
  vec3 dpdu;
  vec3 dpdv;
  vec2 uv;
  uint material_id;
  float distance;
};

struct SampledLight
{
  Spectrum emission;
  vec3 position;
  float pdf;
  vec3 wiW;
  float distance;
  vec3 rand_sample;
  uint light_index;
};

struct LightPhoton
{
  Spectrum emission;
  vec3 origin;
  float pdf_pos;
  vec3 direction;
  float pdf_dir;
  vec3 normal;
  uint light_id;
  vec2 rand_sample;
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
  uint material_id;
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
  uint material_id;
  vec3 rand_sample;
  float pdf;
};

struct BsdfPdf
{
  ShadingSpace shading;
  vec3 woW;
  float pdf;
  vec3 wiW;
  uint material_id;
  vec2 uv;
};

#endif
