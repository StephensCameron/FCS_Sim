#pragma once

typedef struct Coordinate {
    double x;
    double y;
    double z;
} coord_t; 

coord_t* new_random_coord(double bound);
double get_random_double();
coord_t* new_coord(double x, double y, double z);
void timestep_coord(double delta_t, coord_t* coord);
void check_coord_boundary(coord_t* coord, double boundary);
double walk_axis(double D, double delta_t);