#ifndef _RANDOM_GLSL_
#define _RANDOM_GLSL_

const uint MANTISSA_MASK = 0x007FFFFFu;
const uint FLOAT_ONE = 0x3F800000u;

uint hash(uint seed)
{
  uint state = seed * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

uint randi(inout uint rng_state) {
  rng_state = hash(rng_state);
  return rng_state;
}

uint srand(uint seed)
{
  return hash(seed);
}

uint srand(float seed)
{
  return hash(floatBitsToUint(seed));
}

uint srand(vec2 seed)
{
  uint x = floatBitsToUint(seed.x);
  uint y = floatBitsToUint(seed.y);
  return hash(x ^ hash(y));
}

uint srand(vec3 seed)
{
  uint x = floatBitsToUint(seed.x);
  uint y = floatBitsToUint(seed.y);
  uint z = floatBitsToUint(seed.z);
  return hash(x ^ hash(y ^ hash(z)));
}

float rand(inout uint state) {
  uint flt = FLOAT_ONE | (randi(state) & MANTISSA_MASK);
  return uintBitsToFloat(flt) - 1.0;
}

vec2 rand2(inout uint state)
{
  return vec2(rand(state), rand(state));
}

vec3 rand3(inout uint state)
{
  return vec3(rand(state), rand(state), rand(state));
}

#endif
