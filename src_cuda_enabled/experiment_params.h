#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Host-side experimental setup for a gpu_gen_timetrace run.
// particle_type_Ds and particle_type_counts_per_second are parallel arrays of
// length num_particle_types; particles are assigned to types round-robin
// (particle i gets type i % num_particle_types) when the trajectory buffer
// is initialized.
typedef struct ExperimentParams {
    int num_particles;
    float boundary;
    float delta_t;
    int num_timesteps;
    float w0;
    float z0;
    int num_particle_types;
    float* particle_type_Ds;
    float* particle_type_counts_per_second;
} experiment_params_t;

#ifdef __cplusplus
}
#endif
