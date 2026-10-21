#pragma once
#include "coord.h"
#include "confocal_volume.h"
#include "diffusion_sim.h"


// 


// A collection of intensities at each delta_t timestep
typedef struct TimeTrace {
    double* time; // array of time points
    double* intensities; // array of intensities at each time point
    double delta_t; // time step size
    int num_timesteps; // number of time steps
    int counts_per_second; // cpms for each particle. temporarily stored in timetrace, should be in coord.
} time_trace_t;

typedef struct trace_autocorrelation {
    double* time_lags; // array of time lags
    double* autocorrelations; // array of autocorrelations at each time lag
    int num_lags; // number of time lags
} trace_autocorrelation_t;


time_trace_t* initialize_time_trace(double delta_t, int num_timesteps, int counts_per_second);
void update_time_trace(time_trace_t* trace, sim_t* sim, confocal_volume_t* volume, int timestep_index);
double calc_intensity_at_coord(coord_t* coord, confocal_volume_t* volume);
double calc_intensity_rate_at_time(sim_t* sim, confocal_volume_t* volume, time_trace_t* trace);

trace_autocorrelation_t* calculate_autocorrelation(time_trace_t* trace, int max_lag);

void free_time_trace(time_trace_t* trace);
void free_trace_autocorrelation(trace_autocorrelation_t* autocorr);