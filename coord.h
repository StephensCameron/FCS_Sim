#pragma once

typedef struct  Coordinate {
    double x;
    double y;
    double z;
} coord_t; 

coord_t new_random_coord(); 
coord_t new_coord(double x, double y, double z);
void timestep_coord(double delta_t, coord_t* coord);
