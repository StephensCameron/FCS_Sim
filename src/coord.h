#pragma once
#include "confocal_volume.h"

typedef struct Coordinate {
    double x;
    double y;
    double z;
} coord_t; 

coord_t* new_random_coord(double bound);
void seed_rng(unsigned int seed);
double get_random_double();
coord_t* new_coord(double x, double y, double z);
void timestep_coord(double D, double delta_t, coord_t* coord);
void check_coord_boundary(coord_t* coord, double boundary);
double walk_axis(double D, double delta_t);

double get_intensity_at_coord(coord_t* coord, confocal_volume_t* volume);