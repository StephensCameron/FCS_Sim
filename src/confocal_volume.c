#include "confocal_volume.h"
#include "math.h"
#include "stdlib.h"


void free_confocal_volume(confocal_volume_t* volume) {
    if (volume != NULL) {
        free(volume);
    }
}