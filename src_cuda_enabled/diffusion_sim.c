// diffusion sim will be a collection of coordinates, and a set of boundary conditions.
// It will also initialize a certain number of coordinates, and then run a simulation of diffusion for a certain number of timesteps.

#include "coord.h"
#include "diffusion_sim.h"
#include "stdlib.h"
#include "math.h"


sim_t* sim_create(int number_particles, double bound, double D){
    //initialize simulation
    sim_t* sim = malloc(sizeof(sim_t));
    sim->number_particles = number_particles; 
    sim->boundary = bound;
    sim->D = D;
    sim->list_of_coordinates = malloc(sizeof(coord_t*)*number_particles);
    // initialize the coordinantes at random points
    for (int i=0; i<number_particles; i++){
        sim->list_of_coordinates[i] = new_random_coord(2*bound);
    }
    return sim;
}

void step_sim(sim_t* sim, double delta_t) {
    // loop through each particle in the sim and timestep it
    for (int i = 0; i<sim->number_particles; i++) {
        timestep_coord(sim->D, delta_t, sim->list_of_coordinates[i]);
        check_coord_boundary(sim->list_of_coordinates[i], sim->boundary);
    }
    return; 
}

void sim_destroy(sim_t* sim) {
    for (int i = 0; i<sim->number_particles; i++) {
        free(sim->list_of_coordinates[i]);
    }
    free(sim->list_of_coordinates);
    free(sim);
    return;
}

coord_t** sim_get_coordinates(sim_t* sim) {
    return sim->list_of_coordinates;
}