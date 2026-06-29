#pragma once
#include "coord.h"

typedef struct DiffusionSim{
    coord_t** list_of_coordinates;
    double boundary;
    int number_particles;
} sim_t; 


sim_t* sim_create(int number_particles, double bound);
void sim_destroy(sim_t* sim);
void step_sim(sim_t* sim, double delta_t);
coord_t** sim_get_coordinates(sim_t* sim);