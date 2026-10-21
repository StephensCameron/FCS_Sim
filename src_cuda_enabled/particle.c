#include "particle.h"
#include <stdlib.h>
#include <stdio.h>

#include "coord.h"


particle_t* new_random_particle(double bound, float counts_per_second, double D) {
    particle_t* particle = malloc(sizeof(particle_t));
    particle->D = D;
    particle->counts_per_second = counts_per_second;
    particle->coord = *new_random_coord(bound);

    return particle;
}