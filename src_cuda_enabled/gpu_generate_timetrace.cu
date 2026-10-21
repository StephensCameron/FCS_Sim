#pragma once
#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "time.h"
#include "iostream"

#include "coord.h"
#include "diffusion_sim.h"
#include "confocal_volume.h"
#include "time_trace.h"

#include <cuda_runtime.h>


//nvcc --shared -o DLLs_cuda_enabled/gpu_generate_timetrace.dll src_cuda_enabled/gpu_generate_timetrace.cu src_cuda_enabled/coord.c src_cuda_enabled/diffusion_sim.c src_cuda_enabled/time_trace.c src_cuda_enabled/confocal_volume.c


//// nvcc --shared -o DLLs_cuda_enabled/gpu_generate_timetrace.dll src_cuda_enabled/gpu_generate_timetrace.cu

__global__ void calculate_trajectories(coord_t* traj_buffer, float bound, int num_particles, int num_timesteps, float delta_T, float D) {
    //int n = blockIdx.x * blockDim.x + threadIdx.x;
    int n = threadIdx.x;
    int start_idx = n*num_timesteps;
    int stop_idx = (n+1)*num_timesteps;
    double M_PI = 3.141592653589793;

    // using 64-bit LCG that is seeded by the thread index
    // note that the LCG and box-muller implementations are taken from Numerical Recipes Book
    // I choose to implement my own LCG and Box-Muller transform for generating Gaussian random numbers where I can ensure
    // the independence of random numbers across threads.

    uint64_t unifrnd_LL = uint64_t(n)+1ULL; // seed can not be zero
    double unifrnd;
    // if a 32-bit LCG is needed, check Numerical Recipes 2nd edition page 284-285


    // a potential major problem is the bound check, I use periodic BCs and check with an if statement after updating the coordinates.
    // This might significantly affect speed as it introduces different instructions across threads.



    // for this we only will be using threads of size less than num particles
    if (n < num_particles) {
        for (int i = start_idx; i<stop_idx-1; i++) {

            // update LCG for x coord
            unifrnd_LL = unifrnd_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd = float((double)(uint32_t)(unifrnd_LL >> 32) / (double)UINT32_MAX);
            
            // transform using box-muller to get normally distributed random number
            double gaussian_rnd = sqrt(-2.0 * log(unifrnd)) * cos(2.0 * M_PI * unifrnd);
            
            // apply step to x coordinate
            traj_buffer[i+1].x = traj_buffer[i].x + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for x coordinate
            if (traj_buffer[i+1].x > bound) traj_buffer[i+1].x -= 2.0 * bound;
            if (traj_buffer[i+1].x < -bound) traj_buffer[i+1].x += 2.0 * bound;
            
            // update LCG for y coord
            unifrnd_LL = unifrnd_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd = float((double)(uint32_t)(unifrnd_LL >> 32) / (double)UINT32_MAX);
            gaussian_rnd = sqrt(-2.0 * log(unifrnd)) * cos(2.0 * M_PI * unifrnd);
            traj_buffer[i+1].y = traj_buffer[i].y + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for y coordinate
            if (traj_buffer[i+1].y > bound) traj_buffer[i+1].y -= 2.0 * bound;
            if (traj_buffer[i+1].y < -bound) traj_buffer[i+1].y += 2.0 * bound;

            // update LCG for z coord
            unifrnd_LL = unifrnd_LL * 2862933555777941757ULL + 3037000493ULL;
            unifrnd = float((double)(uint32_t)(unifrnd_LL >> 32) / (double)UINT32_MAX);
            gaussian_rnd = sqrt(-2.0 * log(unifrnd)) * cos(2.0 * M_PI * unifrnd);
            traj_buffer[i+1].z = traj_buffer[i].z + gaussian_rnd * sqrt(2.0 * D * delta_T);
            // apply periodic boundary conditions for z coordinate
            if (traj_buffer[i+1].z > bound) traj_buffer[i+1].z -= 2.0 * bound;
            if (traj_buffer[i+1].z < -bound) traj_buffer[i+1].z += 2.0 * bound;

        }
    }

}

__global__ void calculate_individual_brightnesses(coord_t* traj_buffer, double* intensities_buffer, int num_particles, int brightness, int num_timesteps, float delta_T, float w0, float z0){
    // Calculate the intensity at a given coordinate based on the confocal volume parameters
    // double r_squared = coord->x * coord->x + coord->y * coord->y;
    // double z_squared = coord->z * coord->z;

    // Gaussian beam profile
    // double intensity = exp(-2.0 * r_squared / (volume->w0 * volume->w0)) * exp(-2.0 * z_squared / (volume->z0 * volume->z0));
    

    // traj_buffer: buffer of the trajectories for each particle in sim, a coordinate with each timestep
    // intensities_buffer: all the partial intensities due to each particle 
    // laid out as:
    // buffer[0:num_timesteps-1] is the buffer for particle 0
    // buffer[num_timesteps-1:2*num_timesteops] is the buffer for particle 1

    // calculate global thread index:
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_timesteps*num_particles) {
        double r_squared = traj_buffer[n].x * traj_buffer[n].x + traj_buffer[n].y * traj_buffer[n].y;
        double z_squared = traj_buffer[n].z * traj_buffer[n].z;
        intensities_buffer[n] = exp(-2.0 * r_squared / (w0 * w0)) * exp(-2.0 * z_squared / (z0 * z0));
    }
}


__global__ void sum_timetraces(double* intensities_buffer, double* intensities_sum_array, int num_particles, int brightness, int num_timesteps){
    // get global thread index:
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // note: intensities_sum_array should be initialized to zero

    // would it be faster to first transpose this data so relevant indices are next to each other?
    if (n < num_timesteps){
        for (int i=0; i<num_particles; i++){
        //for (int i=0; i<time_timesteps; i++){
            intensities_sum_array[n] += intensities_buffer[n+i*num_timesteps];
        }
    }
}





// GPU generate timetrace:
// Pseudocode:
extern "C" __declspec(dllexport) time_trace_t* gpu_gen_timetrace(int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, int NUM_TIMESTEPS, int COUNTS_PER_SECOND, float W0, float Z0) {
    srand(time(NULL)); // Seed the random number generator

    sim_t* sim = sim_create(NUM_PARTICLES, BOUNDARY, D);
    // two approaches:
    // A:
    // Generate all particle trajectories, store in a 2d array.
    // then calculate the intensity due to each particle at each time
    // then sum over all individual intensities. 

    // Generate all particle trajectories:
    // first allocate the memory buffer:
    size_t n_elem = (size_t)NUM_PARTICLES * (size_t)NUM_TIMESTEPS;
    coord_t* trajectories_buffer = (coord_t*)calloc(n_elem, sizeof(coord_t));

    

    double gib_traj_buffer = (double)(n_elem * sizeof(coord_t)) / (1024.0 * 1024.0 * 1024.0);
    printf("Size of trajectories buffer: %.6f GiB\n", gib_traj_buffer);
    
    if (trajectories_buffer == NULL) {
    printf("Failed to allocate trajectories_buffer\n");
    return NULL;
}

    // initialize each particles coord:
    for (int i = 0; i<NUM_PARTICLES; i++){
        trajectories_buffer[NUM_TIMESTEPS*i] = *new_random_coord(BOUNDARY);
    }

    // print first 10 initial coordinates for verification
    for (int i = 0; i < 10; i++) {
        printf("Initial coord %d: x: %f, y: %f, z: %f\n", i, trajectories_buffer[i*NUM_TIMESTEPS].x, trajectories_buffer[i*NUM_TIMESTEPS].y, trajectories_buffer[i*NUM_TIMESTEPS].z);
    }

    // now copy this buffer to device
    size_t size_traj_buffer = NUM_PARTICLES*NUM_TIMESTEPS*sizeof(coord_t);
    coord_t* d_traj_buffer;

    cudaMalloc(&d_traj_buffer, size_traj_buffer);
    cudaMemcpy(d_traj_buffer, trajectories_buffer, size_traj_buffer, cudaMemcpyHostToDevice);

    if (d_traj_buffer == NULL) {
        printf("Failed to allocate device memory for trajectories buffer\n");
        return NULL;
    }

    int threadsPerBlock = NUM_PARTICLES; 
    int blocksPerGrid = 1;//placeholder


    time_trace_t* time_trace = initialize_time_trace(DELTA_T, NUM_TIMESTEPS, COUNTS_PER_SECOND);
    if (time_trace == NULL) {
        printf("Failed to initialize time trace\n");
        return NULL;
    }

    // call gpu kernel:
    //(coord_t* traj_buffer, float bound, float num_particles, float num_timesteps, float delta_T)
    // this kernel calculates the trajetories for all particles and stores all particle paths as an array
    calculate_trajectories<<<blocksPerGrid, threadsPerBlock>>>(d_traj_buffer, BOUNDARY, NUM_PARTICLES, NUM_TIMESTEPS, DELTA_T, D);
    
    // initialize the timesteps in time trace buffer before synchronizing with the device
    // not needed for cuda
    // double curr = 0.0;
    // for (int i = 0; i < NUM_TIMESTEPS; i++) {
    //     time_trace->time[i] = curr;
    //     curr += DELTA_T;
    // }

    cudaDeviceSynchronize();

    // now begin the kernel to calculate the brightness from
    double* d_intensities_buffer;
    double* d_intensities_sum_buffer;

    // check for errors after memory allocation
    size_t intensities_bytes = sizeof(double) * (size_t)NUM_PARTICLES * (size_t)NUM_TIMESTEPS;
    size_t intensities_sum_bytes = sizeof(double) * (size_t)NUM_TIMESTEPS;

    cudaMalloc(&d_intensities_buffer, intensities_bytes);
    printf("Allocated device memory for intensities buffer: %.6f GiB\n", (double)intensities_bytes / (1024.0 * 1024.0 * 1024.0));
    cudaMalloc(&d_intensities_sum_buffer, intensities_sum_bytes);
    printf("Allocated device memory for intensities sum buffer: %.6f GiB\n", (double)intensities_sum_bytes / (1024.0 * 1024.0 * 1024.0));

    if (d_intensities_buffer == NULL || d_intensities_sum_buffer == NULL) {
        printf("Error allocating memory on the GPU\n");
        return NULL;
    }

    cudaMemcpy(d_intensities_sum_buffer, time_trace->intensities, sizeof(double)*NUM_TIMESTEPS, cudaMemcpyHostToDevice);
    // dont think i need to initialize the intensities sum buffer

    // change the blocks per grid and threads per block for the brightness calculation kernel as needed
    threadsPerBlock = 256;
    // calculate_individual_brightnesses indexes over all particles' timesteps, so the grid must cover
    // NUM_TIMESTEPS*NUM_PARTICLES threads, not just NUM_TIMESTEPS
    int brightnessBlocksPerGrid = (NUM_TIMESTEPS*NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;

    // call the brightness calculation kernel
    //__global__ void calculate_individual_brightnesses(coord_t* traj_buffer, double* intensities_buffer, int num_particles, int brightness, int num_timesteps, float delta_T, float w0, float z0){
    calculate_individual_brightnesses<<<brightnessBlocksPerGrid, threadsPerBlock>>>(d_traj_buffer, d_intensities_buffer, NUM_PARTICLES, COUNTS_PER_SECOND, NUM_TIMESTEPS, DELTA_T, W0, Z0);
    cudaDeviceSynchronize();

    // sum_timetraces indexes over NUM_TIMESTEPS threads only
    blocksPerGrid = (NUM_TIMESTEPS + threadsPerBlock - 1) / threadsPerBlock;

    
    // done with trajectories buffer, can free now
    //cudaFree(d_traj_buffer);

    // sum the intensities across all particles for each timestep
    //double* intensities_buffer, double* intensities_sum_array, int num_particles, int brightness, int num_timesteps
    sum_timetraces<<<blocksPerGrid, threadsPerBlock>>>(d_intensities_buffer, d_intensities_sum_buffer, NUM_PARTICLES, COUNTS_PER_SECOND, NUM_TIMESTEPS);
    cudaDeviceSynchronize();

    // transfer trajectories back to host
    cudaMemcpy(trajectories_buffer, d_traj_buffer, size_traj_buffer, cudaMemcpyDeviceToHost);

    //print first 10 coords:
    for (int i =0; i<10; i++){
        printf("x: %f, y: %f, z: %f\n", trajectories_buffer[i].x, trajectories_buffer[i].y, trajectories_buffer[i].z);
    }


    // Current GPU path is a smoke-test stub: return a defined null pointer.
    cudaMemcpy(time_trace->intensities, d_intensities_sum_buffer, sizeof(double)*NUM_TIMESTEPS, cudaMemcpyDeviceToHost);


    // free all memory from GPU
    cudaFree(d_traj_buffer);
    cudaFree(d_intensities_buffer);
    cudaFree(d_intensities_sum_buffer);
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
//gen_timetrace(int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, int NUM_TIMESTEPS, int COUNTS_PER_SECOND, float W0, float Z0)
gpu_gen_timetrace(10, 10,  100, 0.00001, 100, 100, 1.0, 2.0);
return 0;
};