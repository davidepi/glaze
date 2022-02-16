struct ShadingSpace
{
  vec3 s;
  vec3 t;
  vec3 n;
};

ShadingSpace new_shading_space(in vec3 dpdu, in vec3 normal)
{
  vec3 s = normalize(dpdu);
  vec3 t = cross(normal, s);
  return ShadingSpace(s, t, normal);
}

vec3 to_world_space(in vec3 vector, in ShadingSpace matrix)
{
  vec3 ret = vec3(matrix.s.x*vector.x+matrix.t.x*vector.y+matrix.n.x*vector.z,
                  matrix.s.y*vector.x+matrix.t.y*vector.y+matrix.n.y*vector.z,
                  matrix.s.z*vector.x+matrix.t.z*vector.y+matrix.n.z*vector.z);
  return normalize(ret);
}

vec3 to_shading_space(in vec3 w, in ShadingSpace matrix)
{
  vec3 ret = vec3(dot(w, matrix.s), dot(w, matrix.t), dot(w, matrix.n));
  return normalize(ret);
}
