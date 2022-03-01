// Algorithm from this file are from the following papers:
//
// - lambda, D, and G are from "Understanding the Masking-Shadowing Function in 
// Microfacet-Based BRDFs" of Heitz et al.
// https://jcgt.org/published/0003/02/03/paper.pdf (Paper0)
//
// - sampling the GGX distribution is from "Importance Sampling Microfacet-Based
// BSDFs using the Distribution of Visible Normals" again from Heitz et al.
// https://hal.inria.fr/hal-00996995v1/document (Paper1)
// https://hal.inria.fr/hal-00996995v1/file/supplemental1.pdf (Paper2)
//
// All these algorithms use a smith model for masking.
//
#ifndef _MICROFACETS_GLSL_
#define _MICROFACETS_GLSL_

#include "constants.glsl"
#include "raytrace_commons.glsl"

// p22 for isotropic ggx with alpha = 1.0
// the sample wh will stretch the vector to account for roughness
// from algorithm 3 of Paper2
vec2 ggx_sample_p22(in float cost, in vec2 rand_sample)
{
  if(cost>0.999)
  {
    // special case (normal incidence)
    float r = sqrt(rand_sample.x/(1.0-rand_sample.x));
    float phi = TWO_PI*rand_sample.y;
    return vec2(r*cos(phi), r*sin(phi));
  }
  else
  {
    float cos2t = cost*cost;
    float sin2t = max(0.0, 1.0-cos2t);
    float tan2t = CHECKINF(sin2t/cos2t);
    float tant = sqrt(tan2t);
    float a2 = 1.0/tan2t;
    float G1 = 2.0/(1.0+sqrt(1.0+1.0/a2));
    float A = 2.0*rand_sample.x/G1-1.0;
    float B = tant;
    float invA2m1 = 1.0/(A*A-1.0);
    float sqrt_term = sqrt(max(0.0,B*B*invA2m1*invA2m1-(A*A-B*B)*invA2m1));
    float sx1 = B*invA2m1-sqrt_term;
    float sx2 = B*invA2m1+sqrt_term;
    float sx = (A<0.0 || sx2>1.0/tant)?sx1:sx2;
    float stepval = step(0.5, rand_sample.y);
    float s = mix(1.0, -1.0, stepval);
    float u = mix(2.0*(rand_sample.y-0.5), 2.0*(0.5-rand_sample.y), stepval);
    float z = (u*(u*(u*-0.3657289+0.7902350)-0.4249658)+0.0001529)/
              (u*(u*(u*(u*0.1695078-0.3972035)-0.2325005)+1.0)-0.5398259);
    float sy = s*z*sqrt(1.0+sx*sx);
    return vec2(sx, sy);
  }
}

float ggx_d(in vec3 wh, in vec2 a)
{
  float cost = wh.z;
  float cos2t = cost*cost;
  float cos4t = cos2t*cos2t;
  float sin2t = max(0.0, 1.0-cos2t);
  float tan2t = sin2t/cos2t;
  float cos2p = wh.x*wh.x/sin2t;
  float sin2p = wh.y*wh.y/sin2t;
  float eplus1 = 1.0+((cos2p/(a.x*a.x)+sin2p/(a.y*a.y))*tan2t);
  float d = 1.0/(PI*a.x*a.y*cos4t*eplus1*eplus1);
  return isinf(tan2t)?0.0:d;
}

float ggx_lambda(in vec3 v, in vec2 a)
{
  float cost = v.z;
  float cos2t = cost*cost;
  float sin2t = max(0.0, 1.0-cos2t);
  float tan2t = sin2t/cos2t;
  float cos2p = max(0.0, v.x*v.x/sin2t);
  float sin2p = max(0.0, v.y*v.y/sin2t);
  float alpha2 = cos2p*a.x*a.x+sin2p*a.y*a.y;
  float lambda = (-1.0+sqrt(1.0+tan2t*alpha2))*0.5;
  return isinf(tan2t)?0.0:lambda;
}

float ggx_g(in vec3 wo, in vec3 wi, in vec2 a)
{
  return 1.0/(1.0+ggx_lambda(wo, a)+ggx_lambda(wi, a));
}

float ggx_g1(in vec3 v, in vec2 a)
{
  return 1.0/(1.0+ggx_lambda(v, a));
}

float ggx_pdf(in float d, in vec2 a, in vec3 wo, in vec3 wh)
{
  // TODO: the ggx_g1 term should be wo, not wh. dunno why but wo generates A LOT of fireflies everywhere, wh instead is fine.
  // I checked the math for two days straight, still can't figure out the problem.
  return d*ggx_g1(wh, a)*abs(dot(wo, wh))/abs(wh.z);
}

// Algorithm described in Paper1
vec3 ggx_sample_wh(in vec3 wo, in vec2 rand_sample, in vec2 a)
{
  float flip = sign(wo.z);
  vec3 wi = flip*wo;
  // stretch
  vec3 wi_stretched = normalize(vec3(wi.x*a.x, wi.y*a.y, wi.z));
  float cost = wi_stretched.z;
  // compute isotropic p22
  vec2 slope = ggx_sample_p22(cost, rand_sample);
  // rotate
  float cos2t = cost*cost;
  float sin2t = max(0.0, 1.0-cos2t);
  float cosp = sqrt(wi_stretched.x*wi_stretched.x/sin2t);
  float sinp = sqrt(wi_stretched.y*wi_stretched.y/sin2t);
  float sx = cosp*slope.x-sinp*slope.y;
  float sy = sinp*slope.x+cosp*slope.y;
  // undo stretching
  return flip*normalize(vec3(-a.x*sx, -a.y*sy, 1.0));
}

vec2 to_anisotropic(in float a, in float anisotropy)
{
  return vec2(a*(1.0+anisotropy), a*(1.0-anisotropy));
}

#endif
