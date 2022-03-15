// Automatically generated by build.rs from src/vulkan/raytrace_structures.rs
// This file will be automatically re-generated by cargo

#ifndef _RAYTRACE_STRUCTURES_GLSL_
#define _RAYTRACE_STRUCTURES_GLSL_


struct RTFrameData
{
  uint seed;
  uint lights_no;
  vec2 pixel_offset;
  float scene_radius;
  float exposure;
};

struct RTInstance
{
  uint index_offset;
  uint index_count;
  uint material_id;
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
  uint is_specular;
};

struct RTLight
{
  vec4 color0;
  vec4 color1;
  vec4 color2;
  vec4 color3;
  vec4 pos;
  vec4 dir;
  uint shader;
};

#endif

