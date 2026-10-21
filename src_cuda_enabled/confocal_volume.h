#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct confocal_volume{
    double w0; // beam waist
    double z0; // height
} confocal_volume_t;


void free_confocal_volume(confocal_volume_t* volume);

#ifdef __cplusplus
}
#endif