#include "coord.h"
#include "stdlib.h"
#include "math.h"

coord_t* new_coord(double x, double y, double z) {
    coord_t* coord = malloc(sizeof(coord_t));
    coord->x = x;
    coord->y = y;
    coord->z = z;
    return coord;
}

double get_random_double() {
    // 1. rand() / RAND_MAX gives [0.0, 1.0]
    // 2. Multiplying by 2.0 expands the span to 2 units [0.0, 2.0]
    // 3. Subtracting 1.0 shifts the range to [-1.0, 1.0]
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}


coord_t* new_random_coord(double bound) {
    coord_t *rand_coord = malloc(sizeof(coord_t));
    rand_coord->x = ((double)rand() / RAND_MAX) * bound - (bound/2);
    rand_coord->y = ((double)rand() / RAND_MAX) * bound - (bound/2);
    rand_coord->z = ((double)rand() / RAND_MAX) * bound - (bound/2);
    return rand_coord;
}


double walk_axis(double D, double delta_t) {
    // Generate a random double between 0.0 and 1.0
    double scale = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    return sqrt(D * delta_t * 2) * scale; 
}

void timestep_coord(double D, double delta_t, coord_t* coord) {
    coord->x = coord->x + walk_axis(D, delta_t);
    coord->y = coord->y + walk_axis(D, delta_t);
    coord->z = coord->z + walk_axis(D, delta_t);
}

void check_coord_boundary(coord_t* coord, double boundary) {
    // apply a boundary check for a cube of size 2*boundary centered at the origin
    // This allows us to have a cube with corners at (-boundary, -boundary, -boundary) and (boundary, boundary, boundary)
    // implement a period ic boundary condition, where if a coordinate goes beyond the boundary, it wraps around to the other side
    if (coord->x < -boundary) {
        coord->x = 2 * boundary + coord->x; // wrap around to the other side
    } else if (coord->x > boundary) {
        coord->x = -2 * boundary + coord->x; // wrap around to the other side
    }

    if (coord->y < -boundary) {
        coord->y = 2 * boundary + coord->y; // wrap around to the other side
    } else if (coord->y > boundary) {
        coord->y = -2 * boundary + coord->y; // wrap around to the other side
    }

    if (coord->z < -boundary) {
        coord->z = 2 * boundary + coord->z; // wrap around to the other side
    } else if (coord->z > boundary) {
        coord->z = -2 * boundary + coord->z; // wrap around to the other side
    }
}

double get_intensity_at_coord(coord_t* coord, confocal_volume_t* volume) {
    // Calculate the intensity at a given coordinate based on the confocal volume parameters
    double r_squared = coord->x * coord->x + coord->y * coord->y;
    double z_squared = coord->z * coord->z;

    // Gaussian beam profile
    double intensity = exp(-2.0 * r_squared / (volume->w0 * volume->w0)) * exp(-2.0 * z_squared / (volume->z0 * volume->z0));
    
    return intensity;
}