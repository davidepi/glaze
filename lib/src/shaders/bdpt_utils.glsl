#ifndef _BDPT_UTILS_GLSL_
#define _BDPT_UTILS_GLSL_

#include "hit.glsl"
#include "raytrace_structures.glsl"

void surface_hit_to_vertex(in SurfaceHit sh, out BDPTPathVertex vert)
{
  vert.distance = sh.distance;
  vert.miss = sh.miss;
  vert.ids = sh.ids;
  vert.attribs = sh.attribs;
}

void vertex_to_surface_hit(in BDPTPathVertex vert, out SurfaceHit sh)
{
  sh.distance = vert.distance;
  sh.miss = vert.miss;
  sh.ids = vert.ids;
  sh.attribs = vert.attribs;
}

#endif
