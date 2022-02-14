#include "spectrum.glsl"

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
  float power;
  uint shader;
};
