#pragma once
#include "coord.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DiffusionSim{
    coord_t** list_of_coordinates;
    double boundary;
    double D;
    int number_particles;
} sim_t;


sim_t* sim_create(int number_particles, double bound, double D);
void sim_destroy(sim_t* sim);
void step_sim(sim_t* sim, double delta_t);
coord_t** sim_get_coordinates(sim_t* sim);
void step_sim_cuda(sim_t* sim, double delta_t, int num_timesteps);

#ifdef __cplusplus
}
#endif
