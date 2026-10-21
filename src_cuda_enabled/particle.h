#pragma once
#include "coord.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Particle {
    float D;
    float counts_per_second;
    coord_t coord;
} particle_t;


particle_t* new_random_particle(double bound, float counts_per_second, double D);



#ifdef __cplusplus
}
#endif

