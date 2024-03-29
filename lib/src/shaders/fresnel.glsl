#ifndef _FRESNEL_GLSL_
#define _FRESNEL_GLSL_

#include "spectrum.glsl"

// ior2abs2 is ior*ior+absorption*absorption
Spectrum fresnel_conductor(in float cosin, in Spectrum ior, in Spectrum ior2abs2)
{
  float cosin2 = cosin*cosin;
  Spectrum etacosin2 = mul(ior, cosin*2.0);
  Spectrum etacosin2plus = add(etacosin2, cosin2);
  Spectrum etacosin2plusplus = add(etacosin2, 1.0);
  Spectrum rperpsq = div(sub(ior2abs2,etacosin2plus),add(ior2abs2,etacosin2plus));
  Spectrum tmp = mul(ior2abs2, cosin2);
  Spectrum rparsq = div(sub(tmp, etacosin2plusplus),add(tmp, etacosin2plusplus));
  return div(add(rperpsq, rparsq), 2.0);
}

float fresnel_dielectric(in float costi, in float etai, in float etat)
{
  float sin2ti = max(0.0, 1.0-costi*costi);
  float sin2tt = etai*etai/(etat*etat) * sin2ti;
  if(sin2tt >= 1.0)
  {
    return 1.0;
  }
  float costt = sqrt(max(0.0, 1.0-sin2tt));
  float etatcostt = etat*costt;
  float etatcosti = etat*costi;
  float etaicosti = etai*costi;
  float etaicostt = etai*costt;
  float rparl = (etatcosti - etaicostt) / (etatcosti + etaicostt);
  float rperp = (etaicosti - etatcostt) / (etaicosti + etatcostt);
  return (rparl*rparl+rperp*rperp)/2.0;
}

#endif
