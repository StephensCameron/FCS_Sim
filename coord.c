#include "coord.h"
#include "stdlib.h"
#include "math.h"

coord_t new_coord(double x, double y, double z) {
    coord_t coord = {.x = x, .y = y, .z = z};
    return coord;
}

double get_random_double() {
    // 1. rand() / RAND_MAX gives [0.0, 1.0]
    // 2. Multiplying by 2.0 expands the span to 2 units [0.0, 2.0]
    // 3. Subtracting 1.0 shifts the range to [-1.0, 1.0]
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}


double step_number(double D, double delta_t) {
    // Generate a random double between 0.0 and 1.0
    double scale = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    return sqrt(D * delta_t * 2) * scale; 
}

void timestep_coord(double delta_t, coord_t* coord) {
    double D = 1.0;
    coord->x = coord->x + step_number(D, delta_t);
    coord->y = coord->y + step_number(D, delta_t);
    coord->z = coord->z + step_number(D, delta_t);
}

