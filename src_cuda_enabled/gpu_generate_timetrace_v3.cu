#pragma once
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "time.h"
#include "iostream"


#include "particle.h"
#include "coord.h"
#include "diffusion_sim.h"
#include "confocal_volume.h"
#include "time_trace.h"
#include "experiment_params.h"

#include <cuda_runtime.h>


//nvcc --shared --cudart hybrid -o DLLs_cuda_enabled/gpu_generate_timetrace.dll src_cuda_enabled/gpu_generate_timetrace_v3.cu src_cuda_enabled/particle.c src_cuda_enabled/coord.c src_cuda_enabled/diffusion_sim.c src_cuda_enabled/time_trace.c src_cuda_enabled/confocal_volume.c


__global__ void calculate_trajectories(particle_t* traj_buffer, float* intensities_buffer, int* seed_arr, float bound, int num_particles, int num_timesteps, float delta_T, float w0_squared, float z0_squared) {
    //int n = blockIdx.x * blockDim.x + threadIdx.x;
    int n = threadIdx.x;
    int start_idx = n*num_timesteps;
    int stop_idx = (n+1)*num_timesteps;
    double M_PI = 3.141592653589793;

    // for this we only will be using threads of size less than num particles
    if (n < num_particles) {

        float D = traj_buffer[n].D;
        float counts_per_second = traj_buffer[n].counts_per_second;

        // using 64-bit LCG that is seeded by the per-particle seed drawn from the host RNG
        // note that the LCG and box-muller implementations are taken from Numerical Recipes Book
        // I choose to implement my own LCG and Box-Muller transform for generating Gaussian random numbers where I can ensure
        // the independence of random numbers across threads.

        uint64_t unif1_LL = (uint64_t)seed_arr[n] + 1ULL; // seed can not be zero
        uint64_t unif2_LL = 2ULL*(uint64_t)seed_arr[n] + 1ULL;
        double unifrnd_1;
        double unifrnd_2;
        // if a 32-bit LCG is needed, check Numerical Recipes 2nd edition page 284-285


        // a potential major problem is the bound check, I use periodic BCs and check with an if statement after updating the coordinates.
        // This might significantly affect speed as it introduces different instructions across threads.




        for (int i = start_idx; i<stop_idx; i++) {
            double r_squared = traj_buffer[n].coord.x * traj_buffer[n].coord.x + traj_buffer[n].coord.y * traj_buffer[n].coord.y;
            double z_squared = traj_buffer[n].coord.z * traj_buffer[n].coord.z;
            intensities_buffer[i] = counts_per_second * exp(-2.0 * r_squared / (w0_squared)) * exp(-2.0 * z_squared / (z0_squared));

            // update LCG for x coord
            unif1_LL = unif1_LL * 2862933555777941757ULL + 3037000493ULL;
            unif2_LL = unif2_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd_1 = float((double)(uint32_t)(unif2_LL >> 32) / (double)UINT32_MAX);
            unifrnd_2 = float((double)(uint32_t)(unif1_LL >> 32) / (double)UINT32_MAX);

            // transform using box-muller to get normally distributed random number

            // this box-muller implementation is wrong
            double gaussian_rnd = sqrt(-2.0 * log(unifrnd_1)) * cos(2.0 * M_PI * unifrnd_2);

            // apply step to x coordinate
            traj_buffer[n].coord.x = traj_buffer[n].coord.x + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for x coordinate
            if (traj_buffer[n].coord.x > bound) traj_buffer[n].coord.x -= 2.0 * bound;
            if (traj_buffer[n].coord.x < -bound) traj_buffer[n].coord.x += 2.0 * bound;

            // update LCG for y coord
            unif1_LL = unif1_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd_1 = float((double)(uint32_t)(unif1_LL >> 32) / (double)UINT32_MAX);
            unif2_LL = unif2_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd_2 = float((double)(uint32_t)(unif2_LL >> 32) / (double)UINT32_MAX);
            gaussian_rnd = sqrt(-2.0 * log(unifrnd_1)) * cos(2.0 * M_PI * unifrnd_2);
            traj_buffer[n].coord.y = traj_buffer[n].coord.y + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for y coordinate
            if (traj_buffer[n].coord.y > bound) traj_buffer[n].coord.y -= 2.0 * bound;
            if (traj_buffer[n].coord.y < -bound) traj_buffer[n].coord.y += 2.0 * bound;

            // update LCG for z coord
            unif1_LL = unif1_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd_1 = float((double)(uint32_t)(unif1_LL >> 32) / (double)UINT32_MAX);
            unif2_LL = unif2_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd_2 = float((double)(uint32_t)(unif2_LL >> 32) / (double)UINT32_MAX);
            gaussian_rnd = sqrt(-2.0 * log(unifrnd_1)) * cos(2.0 * M_PI * unifrnd_2);
            traj_buffer[n].coord.z = traj_buffer[n].coord.z + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for z coordinate
            if (traj_buffer[n].coord.z > bound) traj_buffer[n].coord.z -= 2.0 * bound;
            if (traj_buffer[n].coord.z < -bound) traj_buffer[n].coord.z += 2.0 * bound;

        }
    seed_arr[n] = int((uint32_t)(unif2_LL >> 32)); // increment seed for next execution
    }
}



__global__ void sum_timetraces(float* intensities_buffer, double* intensity_trace,int MAX_TIMESTEPS, int exec_number, int num_particles, int num_timesteps){
    // get global thread index:
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // note: intensities_sum_array should be initialized to zero

    // would it be faster to first transpose this data so relevant indices are next to each other?
    if ((n + exec_number * MAX_TIMESTEPS) < num_timesteps){
        for (int i=0; i<num_particles; i++){
            intensity_trace[exec_number*MAX_TIMESTEPS + n] += intensities_buffer[n+i*MAX_TIMESTEPS];
        }
    }
}


// GPU generate timetrace:
extern "C" __declspec(dllexport) time_trace_t* gpu_gen_timetrace(experiment_params_t* params) {

    // ============================================================
    // 1. Definitions: host-side setup, sizes, and allocations
    // ============================================================
    int NUM_PARTICLES = params->num_particles;
    float BOUNDARY = params->boundary;
    float DELTA_T = params->delta_t;
    int NUM_TIMESTEPS = params->num_timesteps;
    float W0 = params->w0;
    float Z0 = params->z0;
    int num_particle_types = params->num_particle_types;
    float* particle_type_Ds = params->particle_type_Ds;
    float* particle_type_counts_per_second = params->particle_type_counts_per_second;

    srand(time(NULL)); // Seed the random number generator

    // Trajectory and intensity calculation will be capped at 500_000 timesteps before requiring a loop to reduce size of buffers needed
    const int MAX_TIMESTEPS = (NUM_TIMESTEPS > 50000) ? 50000 : NUM_TIMESTEPS;
    const int NUM_EXECUTIONS = (NUM_TIMESTEPS + MAX_TIMESTEPS - 1) / MAX_TIMESTEPS;


    // per-particle RNG seeds, drawn from the (now randomized) host RNG
    int* seed_arr = (int*)malloc(NUM_PARTICLES * sizeof(int));
    if (seed_arr == NULL) {
        printf("Failed to allocate seed array\n");
        return NULL;
    }
    for (int i = 0; i < NUM_PARTICLES; i++) {
        seed_arr[i] = rand();
    }
    int* d_seed_arr;

    // particle positions: only the current position per particle is kept (see calculate_trajectories)
    size_t n_elem = (size_t)NUM_PARTICLES;
    particle_t* trajectories_buffer = (particle_t*)calloc(n_elem, sizeof(particle_t));
    if (trajectories_buffer == NULL) {
        printf("Failed to allocate trajectories_buffer\n");
        return NULL;
    }
    double gib_traj_buffer = (double)(n_elem * sizeof(particle_t)) / (1024.0 * 1024.0 * 1024.0);
    printf("Size of trajectories buffer: %.6f GiB\n", gib_traj_buffer);

    // initialize each particle's coord, assigning particle types round-robin
    // so an arbitrary number of particle types (each with its own D and
    // brightness) can be mixed into the same simulation:
    for (int i = 0; i<NUM_PARTICLES; i++){
        int type_idx = i % num_particle_types;
        trajectories_buffer[i] = *new_random_particle(BOUNDARY, particle_type_counts_per_second[type_idx], particle_type_Ds[type_idx]);
    }



    size_t size_traj_buffer = NUM_PARTICLES*sizeof(particle_t);
    particle_t* d_traj_buffer;

    int threadsPerBlock = NUM_PARTICLES;
    int blocksPerGrid = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // counts_per_second on the trace is just metadata (not used in the kernels, which
    // read brightness per-particle from traj_buffer); record the first particle type's.
    time_trace_t* time_trace = initialize_time_trace(DELTA_T, NUM_TIMESTEPS, (int)particle_type_counts_per_second[0]);
    if (time_trace == NULL) {
        printf("Failed to initialize time trace\n");
        return NULL;
    }

    float* d_intensities_buffer;
    double* d_intensities_sum_buffer;
    size_t intensities_bytes = sizeof(float) * NUM_PARTICLES * MAX_TIMESTEPS;
    size_t intensities_sum_bytes = sizeof(double) * MAX_TIMESTEPS;

    float W0_SQUARED = W0 * W0;
    float Z0_SQUARED = Z0 * Z0;

    // ============================================================
    // 2. Device allocations and host<->device transfers
    // ============================================================
    cudaMalloc(&d_seed_arr, sizeof(int)*NUM_PARTICLES);
    cudaMemcpy(d_seed_arr, seed_arr, sizeof(int)*NUM_PARTICLES, cudaMemcpyHostToDevice);

    cudaMalloc(&d_traj_buffer, size_traj_buffer);
    cudaMemcpy(d_traj_buffer, trajectories_buffer, size_traj_buffer, cudaMemcpyHostToDevice);
    if (d_traj_buffer == NULL) {
        printf("Failed to allocate device memory for trajectories buffer\n");
        return NULL;
    }

    cudaMalloc(&d_intensities_buffer, intensities_bytes);
    printf("Allocated device memory for intensities buffer: %.6f GiB\n", (double)intensities_bytes / (1024.0 * 1024.0 * 1024.0));
    cudaMalloc(&d_intensities_sum_buffer, sizeof(double) * NUM_TIMESTEPS);
    printf("Allocated device memory for intensities sum buffer: %.6f GiB\n", (double)(sizeof(double) * NUM_TIMESTEPS) / (1024.0 * 1024.0 * 1024.0));
    if (d_intensities_buffer == NULL || d_intensities_sum_buffer == NULL) {
        printf("Error allocating memory on the GPU\n");
        return NULL;
    }

    cudaMemcpy(d_intensities_sum_buffer, time_trace->intensities, sizeof(double)*NUM_TIMESTEPS, cudaMemcpyHostToDevice);

    // ============================================================
    // 3. Kernels: trajectory generation, then summation
    // ============================================================
    
    for (int exec = 0; exec < NUM_EXECUTIONS; exec++) {
        blocksPerGrid = 1;
        calculate_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_traj_buffer, d_intensities_buffer, d_seed_arr, BOUNDARY, NUM_PARTICLES, MAX_TIMESTEPS, DELTA_T, W0_SQUARED, Z0_SQUARED);
        cudaDeviceSynchronize();
        // sum_timetraces indexes over NUM_TIMESTEPS threads only
        blocksPerGrid = (MAX_TIMESTEPS + threadsPerBlock - 1) / threadsPerBlock;
        sum_timetraces<<<blocksPerGrid, threadsPerBlock>>>(d_intensities_buffer, d_intensities_sum_buffer, MAX_TIMESTEPS, exec, NUM_PARTICLES, NUM_TIMESTEPS);
        cudaDeviceSynchronize();

    }





    cudaMemcpy(time_trace->intensities, d_intensities_sum_buffer, sizeof(double)*NUM_TIMESTEPS, cudaMemcpyDeviceToHost);
    
    cudaFree(d_traj_buffer);
    cudaFree(d_seed_arr);
    cudaFree(d_intensities_buffer);
    cudaFree(d_intensities_sum_buffer);
    cudaDeviceSynchronize();

    return time_trace;
}



// CPU serial generate timetrace
extern "C" __declspec(dllexport) time_trace_t* gen_timetrace(int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, float TOTAL_TIME_LENGTH, int COUNTS_PER_SECOND, float W0, float Z0) {
    srand(time(NULL)); // Seed the random number generator

    sim_t* sim = sim_create(NUM_PARTICLES, BOUNDARY, D);

    confocal_volume_t* volume = (confocal_volume_t*) malloc(sizeof(confocal_volume_t));
    volume->w0 = W0;
    volume->z0 = Z0;

    int num_timesteps = (int)(TOTAL_TIME_LENGTH / DELTA_T);
    time_trace_t* trace = initialize_time_trace(DELTA_T, num_timesteps, COUNTS_PER_SECOND);

    for (int t = 0; t < num_timesteps; t++) {
        step_sim(sim, DELTA_T);
        update_time_trace(trace, sim, volume, t);
    }


    return trace;
}


// test method:
int test() {
float type_Ds[1] = { 100.0f };
float type_counts_per_second[1] = { 100.0f };
experiment_params_t params = { 10, 10.0f, 0.00001f, 100, 1.0f, 2.0f, 1, type_Ds, type_counts_per_second };
gpu_gen_timetrace(&params);
return 0;
};
