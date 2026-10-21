#include "time_trace.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


time_trace_t* initialize_time_trace(double delta_t, int num_timesteps, int counts_per_second) {
    time_trace_t* time_trace = malloc(sizeof(time_trace_t));
    time_trace->counts_per_second = counts_per_second;
    time_trace->delta_t = delta_t;
    time_trace->num_timesteps = num_timesteps;
    time_trace->time = calloc(num_timesteps, sizeof(double));
    time_trace->intensities = calloc(num_timesteps, sizeof(double));
    return time_trace;
}

double calc_intensity_at_coord(coord_t* coord, confocal_volume_t* volume) {
    // Calculate the intensity at a given coordinate based on the confocal volume parameters
    double r_squared = coord->x * coord->x + coord->y * coord->y;
    double z_squared = coord->z * coord->z;

    // Gaussian beam profile
    double intensity = exp(-2.0 * r_squared / (volume->w0 * volume->w0)) * exp(-2.0 * z_squared / (volume->z0 * volume->z0));
    
    return intensity;
}

double calc_intensity_rate_at_time(sim_t* sim, confocal_volume_t* volume, time_trace_t* trace) {
    double total_intensity = 0.0;
    for (int i = 0; i < sim->number_particles; i++) {
        total_intensity += calc_intensity_at_coord(sim->list_of_coordinates[i], volume);
    }
    // Convert to counts per second
    double intensity_rate = total_intensity * trace->counts_per_second;
    return intensity_rate;
}

void update_time_trace(time_trace_t* trace, sim_t* sim, confocal_volume_t* volume, int timestep_index) {
    // Updates the time trace based on the current state of the simulation
    trace->time[timestep_index] = timestep_index * trace->delta_t;
    trace->intensities[timestep_index] = calc_intensity_rate_at_time(sim, volume, trace);
}


trace_autocorrelation_t* calculate_autocorrelation(time_trace_t* trace, int max_lag) {
    trace_autocorrelation_t* autocorr = malloc(sizeof(trace_autocorrelation_t));
    autocorr->num_lags = (int)(max_lag / trace->delta_t) + 1;
    autocorr->time_lags = calloc(autocorr->num_lags, sizeof(double));
    if (autocorr->time_lags == NULL) {
        free(autocorr);
        printf("Memory allocation failed for time_lags\n");
        return NULL; // Memory allocation failed
    }
    autocorr->autocorrelations = calloc(autocorr->num_lags, sizeof(double));
    if (autocorr->autocorrelations == NULL) {
        free(autocorr->time_lags);
        free(autocorr);
        printf("Memory allocation failed for autocorrelations\n");
        return NULL; // Memory allocation failed
    }

    // Calculate mean intensity
    double mean_intensity = 0.0;
    for (int i = 0; i < trace->num_timesteps; i++) {
        mean_intensity += trace->intensities[i];
    }
    mean_intensity /= trace->num_timesteps;

    // Calculate autocorrelation
    for (int lag = 0; lag < autocorr->num_lags; lag++) {
        double sum = 0.0;
        for (int t = 0; t < trace->num_timesteps - lag; t++) {
            sum += (trace->intensities[t] - mean_intensity) * (trace->intensities[t + lag] - mean_intensity);
        }
        autocorr->time_lags[lag] = lag * trace->delta_t;
        autocorr->autocorrelations[lag] = sum / (trace->num_timesteps - lag);
    }

    return autocorr;
}



void free_time_trace(time_trace_t* trace) {
    if (trace) {
        free(trace->time);
        free(trace->intensities);
        free(trace);
    }
}

void free_trace_autocorrelation(trace_autocorrelation_t* autocorr) {
    if (autocorr) {
        free(autocorr->time_lags);
        free(autocorr->autocorrelations);
        free(autocorr);
    }
}