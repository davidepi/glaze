#ifndef _MICROFACETS_GLSL_
#define _MICROFACETS_GLSL_

float ggx_iso_g(in vec3 wo, in vec3 wi, in float a2)
{
  float costo = wo.z;
  float costi = wi.z;
  float ggxwo = costi * sqrt(costo*costo*(1.0-a2)+a2);
  float ggxwi = costo * sqrt(costi*costi*(1.0-a2)+a2);
  return 0.5/(ggxwo+ggxwi);
}

float ggx_iso_d(in vec3 wh, in float a)
{
  float cost = wh.z;
  float cos2t = cost*cost;
  float ac = cost*a;
  float k = a / (1.0-cos2t+ac*ac);
  return k*k*INV_PI;
}

vec3 ggx_iso_sample_wh(in vec3 wo, in vec2 sample, in float a2)
{
  float cos2t = (1.0-sample.x)/(sample.x*(a2-1.0)+1.0);
  float cost = sqrt(cos2t);
  float sint = sqrt(1.0-cos2t);
  float phi = TWO_PI*sample.y;
  vec3 wh = vec3(sint*cos(phi), sint*sin(phi), cost);
  return wh*sign(wo.z*wh.z);
}

float ggx_aniso_d(in vec3 wh, in float ax, in float ay)
{
  float cost = wh.z;
  float cos2t = cost*cost;
  float cos4t = cos2t*cos2t;
  float sin2t = 1.0-cos2t;
  float tan2t = sin2t/cos2t;
  float cos2p = (wh.x*wh.x)/sin2t;
  float sin2p = (wh.y*wh.y)/sin2t;
  float eplus1 = 1.0+(cos2p/(ax*ax)+sin2p/(ay*ay)*tan2t);
  return 1.0/(PI*ax*ay*cos4t*eplus1*eplus1);
}

float ggx_aniso_lambda(in vec3 v, in float ax, float ay)
{
  float cost = v.z;
  float cos2t = cost*cost;
  float sin2t = 1.0 - cos2t;
  float tan2t = sin2t/cos2t;
  float cos2phi = v.x*v.x/sin2t;
  float sin2phi = v.y*v.y/sin2t;
  float alpha = sqrt(cos2phi*ax*ax+sin2phi*ay*ay);
  return (-1.0+sqrt(1.0f+(tan2t*alpha*alpha)))*0.5f;
}

float ggx_aniso_g(in vec3 wo, in vec3wi, in float ax, in float ay)
{
  return 1.0/(1.0+ggx_aniso_lambda(wo, ax, ay)+ggx_aniso_lambda(wi, ax, ay));
}

vec3 ggx_aniso_sample_wh(in vec3 wo, in vec2 sample, in float ax, in float ay)
{
  float phi = atan(ay/ax*tan(TWO_PI*sample.y+0.5*PI));
  phi += step(0.5, sample.y)*PI;
  float sinp = sin(phi);
  float cosp = cos(phi);
  float ax2 = ax*ax;
  float ay2 = ay*ay;
  float a2 = 1.0/(cosp*cosp/ax2+sinp*sinp/ay2);
  float tan2t = a2*sample.x/(1.0-sample.x);
  float cost = 1.0/sqrt(1.0+tan2t);
  float sint = sqrtf(1.f-cost*cost);
  vec3 wh = vec3(sint*cosp, sint*sinp, cost);
  return wh*sign(wo.z*wh.z);
}

#endif
