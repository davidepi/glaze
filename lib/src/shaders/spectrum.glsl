#ifndef _SPECTRUM_GLSL_
#define _SPECTRUM_GLSL_

struct Spectrum
{
  vec4 col0;
  vec4 col1;
  vec4 col2;
  vec4 col3;
};

#define SPECTRUM_ONE Spectrum(vec4(1.0),vec4(1.0),vec4(1.0),vec4(1.0))
#define SPECTRUM_ZERO Spectrum(vec4(0.0),vec4(0.0),vec4(0.0),vec4(0.0))

Spectrum spectrum_uniform(float f)
{
  return Spectrum(vec4(f), vec4(f), vec4(f), vec4(f));
}

Spectrum mix(Spectrum a, Spectrum b, float value)
{
  return Spectrum(
    mix(a.col0, b.col0, value),
    mix(a.col1, b.col1, value),
    mix(a.col2, b.col2, value),
    mix(a.col3, b.col3, value)
  );
}

bool is_black(Spectrum sp)
{
  vec4 black = vec4(0.0);
  return sp.col0 == black &&
    sp.col1 == black &&
    sp.col2 == black &&
    sp.col3 == black;
}

float luminance(Spectrum sp)
{
  vec4 y0 = vec4(0.0013635, 0.0104404, 0.0335974, 0.0778085);
  vec4 y1 = vec4(0.1697079, 0.3826326, 0.7333517, 0.9504986);
  vec4 y2 = vec4(0.9897124, 0.8830489, 0.6703505, 0.4343900);
  vec4 y3 = vec4(0.2263996, 0.0935659, 0.0302771, 0.0085307);
  vec4 yvec = sp.col0 * y0 + sp.col1 * y1 + sp.col2 * y2 + sp.col3 * y3;
  float y = yvec.x + yvec.y + yvec.z + yvec.w;
  return y * 0.17557178;
}

vec3 xyz(Spectrum sp)
{

  vec4 x0 = vec4(0.0485479, 0.2486433, 0.3391310, 0.2375931);
  vec4 x1 = vec4(0.0685131, 0.0074336, 0.0859658, 0.3015185);
  vec4 x2 = vec4(0.5851421, 0.8840365, 1.0476296, 0.9172956);
  vec4 x3 = vec4(0.5582481, 0.2484632, 0.0826624, 0.0235659);
  vec4 y0 = vec4(0.0013635, 0.0104404, 0.0335974, 0.0778085);
  vec4 y1 = vec4(0.1697079, 0.3826326, 0.7333517, 0.9504986);
  vec4 y2 = vec4(0.9897124, 0.8830489, 0.6703505, 0.4343900);
  vec4 y3 = vec4(0.2263996, 0.0935659, 0.0302771, 0.0085307);
  vec4 z0 = vec4(0.2318528, 1.2145957, 1.7617404, 1.4557389);
  vec4 z1 = vec4(0.6637067, 0.2402145, 0.0750986, 0.0208246);
  vec4 z2 = vec4(0.0045391, 0.0017035, 0.0009007, 0.0002573);
  vec4 z3 = vec4(3.9191221E-05, 1.9643332E-06, 0.0, 0.0);
  vec4 xvec = sp.col0 * x0 + sp.col1 * x1 + sp.col2 * x2 + sp.col3 * x3;
  vec4 yvec = sp.col0 * y0 + sp.col1 * y1 + sp.col2 * y2 + sp.col3 * y3;
  vec4 zvec = sp.col0 * z0 + sp.col1 * z1 + sp.col2 * z2 + sp.col3 * z3;
  float x = xvec.x + xvec.y + xvec.z + xvec.w;
  float y = yvec.x + yvec.y + yvec.z + yvec.w;
  float z = zvec.x + zvec.y + zvec.z + zvec.w;
  return vec3(x * 0.17557178, y * 0.17557178, z * 0.17557178);
}

vec3 rgb(vec3 xyz)
{
  vec3 rgb;
  rgb.r =  3.240479 * xyz.x - 1.537150 * xyz.y - 0.498535 * xyz.z;
  rgb.g = -0.969256 * xyz.x + 1.875991 * xyz.y + 0.041556 * xyz.z;
  rgb.b =  0.055648 * xyz.x - 0.204043 * xyz.y + 1.057311 * xyz.z;
  return rgb;
}

vec3 rgb(Spectrum sp)
{
  return rgb(xyz(sp));
}

Spectrum add(Spectrum sp, float f)
{
  Spectrum res;
  res.col0 = sp.col0+f;
  res.col1 = sp.col1+f;
  res.col2 = sp.col2+f;
  res.col3 = sp.col3+f;
  return res;
}

Spectrum add(Spectrum sp0, Spectrum sp1)
{
  Spectrum res;
  res.col0 = sp0.col0 + sp1.col0;
  res.col1 = sp0.col1 + sp1.col1;
  res.col2 = sp0.col2 + sp1.col2;
  res.col3 = sp0.col3 + sp1.col3;
  return res;
}

Spectrum sub(Spectrum sp0, Spectrum sp1)
{
  Spectrum res;
  res.col0 = sp0.col0 - sp1.col0;
  res.col1 = sp0.col1 - sp1.col1;
  res.col2 = sp0.col2 - sp1.col2;
  res.col3 = sp0.col3 - sp1.col3;
  return res;
}

Spectrum mul(Spectrum sp, float f)
{
  Spectrum res;
  res.col0 = sp.col0 * f;
  res.col1 = sp.col1 * f;
  res.col2 = sp.col2 * f;
  res.col3 = sp.col3 * f;
  return res;
}

Spectrum mul(Spectrum sp0, Spectrum sp1)
{
  Spectrum res;
  res.col0 = sp0.col0 * sp1.col0;
  res.col1 = sp0.col1 * sp1.col1;
  res.col2 = sp0.col2 * sp1.col2;
  res.col3 = sp0.col3 * sp1.col3;
  return res;
}

Spectrum div(Spectrum sp, float f)
{
  Spectrum res;
  res.col0 = sp.col0 / f;
  res.col1 = sp.col1 / f;
  res.col2 = sp.col2 / f;
  res.col3 = sp.col3 / f;
  return res;
}

Spectrum div(Spectrum sp0, Spectrum sp1)
{
  Spectrum res;
  res.col0 = sp0.col0 / sp1.col0;
  res.col1 = sp0.col1 / sp1.col1;
  res.col2 = sp0.col2 / sp1.col2;
  res.col3 = sp0.col3 / sp1.col3;
  return res;
}

#define GENERATE_COLOR_TO_SPECTRUM                   \
  if(rgb.r <= rgb.g && rgb.r <= rgb.b)               \
  {                                                  \
    res = mul(white, rgb.r);                         \
    if (rgb.g <= rgb.b)                              \
    {                                                \
      res = add(res, mul(cyan, (rgb.g - rgb.r)));    \
      res = add(res, mul(blue, (rgb.b - rgb.g)));    \
    }                                                \
    else                                             \
    {                                                \
      res = add(res, mul(cyan, (rgb.b - rgb.r)));    \
      res = add(res, mul(green, (rgb.g - rgb.b)));   \
    }                                                \
  }                                                  \
  else if(rgb.g <= rgb.r && rgb.g <= rgb.b)          \
  {                                                  \
    res = mul(white, rgb.g);                         \
    if (rgb.r <= rgb.b)                              \
    {                                                \
      res = add(res, mul(magenta, (rgb.r - rgb.g))); \
      res = add(res, mul(blue, (rgb.b - rgb.r)));    \
    }                                                \
    else                                             \
    {                                                \
      res = add(res, mul(magenta, (rgb.b - rgb.g))); \
      res = add(res, mul(red, (rgb.r - rgb.b)));     \
    }                                                \
  }                                                  \
  else                                               \
  {                                                  \
    res = mul(white, rgb.b);                         \
    if (rgb.r <= rgb.g)                              \
    {                                                \
      res = add(res, mul(yellow, (rgb.r - rgb.b)));  \
      res = add(res, mul(green, (rgb.g - rgb.r)));   \
    }                                                \
    else                                             \
    {                                                \
      res = add(res, mul(yellow, (rgb.g - rgb.b)));  \
      res = add(res, mul(red, (rgb.r - rgb.g)));     \
    }                                                \
  }

Spectrum from_surface_color(vec3 rgb)
{
  Spectrum white = Spectrum(
    vec4(1.0619347, 1.0623373, 1.0624330, 1.0624851),
    vec4(1.0622214, 1.0613081, 1.0613059, 1.0618169),
    vec4(1.0624642, 1.0624839, 1.0624682, 1.0625356),
    vec4(1.0624016, 1.0622653, 1.0602665, 1.0600421));
  Spectrum cyan = Spectrum(
    vec4(1.0240953, 1.0245612, 1.0463755, 1.0327847),
    vec4(1.0478429, 1.0535090, 1.0534870, 1.0530664),
    vec4(1.0549103, 0.9429917, 0.3100097, 0.0033711),
    vec4(-0.0048550,0.0018582, 0.0039838, 0.0105072));
  Spectrum magenta = Spectrum(
    vec4(0.9930253, 1.0170691, 1.0143947, 1.0070518),
    vec4(0.8011273, 0.0775935, 0.0032299, -0.0043522),
    vec4(0.0026944, 0.2820553, 0.8570354, 0.9937849),
    vec4(0.9844959, 0.8937981, 0.9495843, 0.9395992));
  Spectrum yellow = Spectrum(
    vec4(-0.0059362,-0.0040293, 0.0346327, 0.1940766),
    vec4(0.4556154, 0.7811726, 1.0163873, 1.0511958),
    vec4(1.0513470, 1.0515278, 1.0512299, 1.0515211),
    vec4(1.0514264, 1.0513103, 1.0507004, 1.0485827));
  Spectrum red = Spectrum(
    vec4(0.1148792, 0.0601411, 0.0040665, 0.0104594),
    vec4(0.0035471,-0.0052706,-0.0062588,-0.0086496),
    vec4(0.0009720, 0.1467938, 0.8584718, 0.9982149),
    vec4(0.9960530, 1.0018494, 0.9959383, 0.9811980));
  Spectrum green = Spectrum(
    vec4(-0.0108655, -0.0103294, -0.0083431, 0.0837942),
    vec4(0.5750078, 0.9511568, 0.9994890, 0.9996808),
    vec4(0.9988160, 0.8861814, 0.3569038, 0.0132560),
    vec4(-0.0050992, -0.0083928, -0.0084414, -0.0047501));
  Spectrum blue = Spectrum(
    vec4(0.9949822, 0.9956945, 0.9998331, 0.9648524),
    vec4(0.6706013, 0.2915789, 0.0446146,-6.7793272E-06),
    vec4(0.0005060, 0.0023498, 0.0006744, 0.0166219),
    vec4(0.0402117, 0.0496045, 0.0435740, 0.0274834));
  Spectrum res;
  GENERATE_COLOR_TO_SPECTRUM
  return mul(res, 0.94);
}

Spectrum from_illuminant_color(vec3 rgb)
{
  Spectrum white = Spectrum(
    vec4(1.1560446, 1.1564162, 1.1567873, 1.1565329),
    vec4(1.1565927, 1.1565410, 1.1472133, 1.1314210),
    vec4(1.0964089, 1.0338718, 0.9652860, 0.9206722),
    vec4(0.9001167, 0.8894007, 0.8808384, 0.8781050));
  Spectrum cyan = Spectrum(
    vec4(1.1352400, 1.1358532, 1.1362707, 1.1359364),
    vec4(1.1361867, 1.1358178, 1.1359519, 1.1354234),
    vec4(1.1224514, 0.8707334, 0.3803442, 0.0512169),
    vec4(-0.0117626, -0.0106069, -0.0069315, -0.0077819));
  Spectrum magenta = Spectrum(
    vec4(1.0765584, 1.0770491, 1.0731253, 1.0796647),
    vec4(1.0024748, 0.4395829, 0.0204297, -0.0015031),
    vec4(-0.0000061, 0.0721516, 0.4807862, 0.9731341),
    vec4(1.0781819, 1.0327506, 1.0495215, 1.0257451));
  Spectrum yellow = Spectrum(
    vec4(0.0001469, -0.0001316, -0.0001677, 0.0895192),
    vec4(0.7482148, 1.0340727, 1.0365779, 1.0367058),
    vec4(1.0365194, 1.0366123, 1.0361321, 1.0144986),
    vec4(0.8293751, 0.6705682, 0.6005960, 0.5827772));
  Spectrum red = Spectrum(
    vec4(0.0571394, 0.0430340, 0.0212607, 0.0010772),
    vec4(0.0005799, -0.0002249, -0.0001201, -0.0001991),
    vec4(0.0127561, 0.1832462, 0.5194882, 0.8212017),
    vec4(0.9626301, 0.9941070, 0.9901806, 0.9827855));
  Spectrum green = Spectrum(
    vec4(0.0064831, 0.0001903, -0.0081060, 0.0481619),
    vec4(0.6672964, 1.0307844, 1.0311600, 1.0265627),
    vec4(1.0363099, 1.0120735, 0.3266872, 0.0033846),
    vec4(0.0081701, 0.0088890, 0.0003663, 0.0009946));
  Spectrum blue = Spectrum(
    vec4(1.0542363, 1.0576206, 1.0581438, 1.0568818),
    vec4(1.0207912, 0.2974275, -0.0014770, -0.0013982),
    vec4(-0.0005919, -0.0010091, -0.0015480, 0.0051107),
    vec4(0.0470549, 0.1282754, 0.1524642, 0.1661573));
  Spectrum res;
  GENERATE_COLOR_TO_SPECTRUM
    return mul(res, 0.86445);
}

#endif
