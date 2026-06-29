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

void timestep_coord(double delta_t, coord_t* coord) {
    double D = 1.0;
    coord->x = coord->x + walk_axis(D, delta_t);
    coord->y = coord->y + walk_axis(D, delta_t);
    coord->z = coord->z + walk_axis(D, delta_t);
}

void check_coord_boundary(coord_t* coord, double boundary) {
    // apply a boundary check for a cube of size 2*boundary centered at the origin
    // This allows us to have a cube with corners at (-boundary, -boundary, -boundary) and (boundary, boundary, boundary)
    if (coord->x < -boundary) {
        coord->x = -2 * boundary;
    } else if (coord->x > boundary) {
        coord->x = 2 * boundary;
    }

    if (coord->y < -boundary) {
        coord->y = -2 * boundary;
    } else if (coord->y > boundary) {
        coord->y = 2 * boundary;
    }

    if (coord->z < -boundary) {
        coord->z = -2 * boundary;
    } else if (coord->z > boundary) {
        coord->z = 2 * boundary;
    }
}

