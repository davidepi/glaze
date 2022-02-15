#include "spectrum.glsl"
#include "shading_space.glsl"

#define INV_PI 0.3183099
#define TWO_PI 6.2831853

struct HitPoint
{
  vec3 point;
  vec3 normal;
  vec3 dpdu;
  vec3 dpdv;
  vec2 uv;
  float distance;
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
  uint bsdf_index;
};

struct BsdfValue
{
  vec3 woW;
  vec3 wiW;
  vec3 normal;
  vec2 uv;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
};

struct BsdfSampleValue
{
  vec3 woW;
  vec3 wiW;
  vec3 normal;
  vec2 uv;
  ShadingSpace shading;
  Spectrum value;
  uint material_index;
  vec2 rand_sample;
  float pdf;
};

struct BsdfPdf
{
  vec3 woW;
  vec3 wiW;
  ShadingSpace shading;
  float pdf;
};
