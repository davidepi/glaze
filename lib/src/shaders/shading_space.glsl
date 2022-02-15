struct ShadingSpace
{
  vec3 s;
  vec3 t;
  vec3 n;
};

ShadingSpace new_shading_space(in vec3 dpdu, in vec3 normal)
{
  vec3 t = normalize(cross(dpdu, normal));
  return ShadingSpace(dpdu, t, normal);
}

vec3 to_world_space(in vec3 s, in ShadingSpace shsp)
{
  vec3 ret = vec3(shsp.s.x*s.x+shsp.t.x*s.y+shsp.n.x*s.z,
                  shsp.s.y*s.x+shsp.t.y*s.y+shsp.n.y*s.z,
                  shsp.s.z*s.x+shsp.t.z*s.y+shsp.n.z*s.z);
  return normalize(ret);
}

vec3 to_shading_space(in vec3 w, in ShadingSpace shsp)
{
  vec3 ret = vec3(dot(w, shsp.s), dot(w, shsp.t), dot(w, shsp.n));
  return normalize(ret);
}
