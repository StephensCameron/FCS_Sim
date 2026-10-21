#include "stdlib.h"
#include "math.h"
#include "stdio.h"
//#include "raylib.h"
#include "time.h"

#include "coord.h"
#include "diffusion_sim.h"
#include "confocal_volume.h"
#include "time_trace.h"

#define WINDOW_W 900 //
#define WINDOW_H 700 //
#define NUM_PARTICLES 100 // number of particles in the simulation
#define BOUNDARY 10.0f // boundary of the simulation box in um
#define DELTA_T 0.0001f // time step size in seconds
#define D 50.0f // diffusion coefficient in um^2/s
#define SPHERE_RADIUS 0.2f // radius of the spheres representing particles in the simulation (N/A in this version)
#define SPHERE_COLOR (Color){ 255, 0, 0, 255 } // color of the spheres representing particles in the simulation (N/A in this version)



// variables for timetrace testing:
#define TOTAL_TIME_LENGTH 300.0f 
#define COUNTS_PER_SECOND 10000
#define MAX_LAG 10.0f
// variables for confocal volume testing:
// #define W0 0.1f
// #define Z0 0.2f

#define W0 1.0f
#define Z0 2.0f






int main() {
    srand(time(NULL)); // Seed the random number generator

    sim_t* sim = sim_create(NUM_PARTICLES, BOUNDARY, D);

    confocal_volume_t* volume = malloc(sizeof(confocal_volume_t));
    volume->w0 = W0;
    volume->z0 = Z0;

    int num_timesteps = (int)(TOTAL_TIME_LENGTH / DELTA_T);
    time_trace_t* trace = initialize_time_trace(DELTA_T, num_timesteps, COUNTS_PER_SECOND);

    for (int t = 0; t < num_timesteps; t++) {
        step_sim(sim, DELTA_T);
        update_time_trace(trace, sim, volume, t);
    }


    // // plot intensity vs time using gnuplot
    // FILE* intensityPipe = popen("gnuplot -persistent", "w");
    // fprintf(intensityPipe, "set title 'Intensity vs Time'\n");
    // fprintf(intensityPipe, "set xlabel 'Time (s)'\n");
    // fprintf(intensityPipe, "set ylabel 'Intensity (counts/s)'\n");
    // fprintf(intensityPipe, "plot '-' with lines title 'Intensity'\n");
    // for (int i = 0; i < trace->num_timesteps; i++) {
    //     fprintf(intensityPipe, "%f %f\n", trace->time[i], trace->intensities[i]);
    // }
    // fprintf(intensityPipe, "e\n");
    // pclose(intensityPipe);

    // calculate autocorrelation
    trace_autocorrelation_t* autocorr = calculate_autocorrelation(trace, MAX_LAG);


    // // print out the autocorrelation results
    // printf("Time Lag (s)\tAutocorrelation\n");
    // for (int i = 0; i < autocorr->num_lags; i++) {
    //     printf("%f\t%f\n", autocorr->time_lags[i], autocorr->autocorrelations[i]);
    // }

    // // plot autocorrelation using gnuplot (log x scale, linear y scale)
    // FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    // fprintf(gnuplotPipe, "set title 'Autocorrelation'\n");
    // fprintf(gnuplotPipe, "set xlabel 'Time Lag (s)'\n");
    // fprintf(gnuplotPipe, "set ylabel 'Autocorrelation'\n");
    // fprintf(gnuplotPipe, "set logscale x\n");
    // fprintf(gnuplotPipe, "plot '-' with lines title 'Autocorrelation'\n");
    // for (int i = 0; i < autocorr->num_lags; i++) {
    //     fprintf(gnuplotPipe, "%f %f\n", autocorr->time_lags[i], autocorr->autocorrelations[i]);
    // }
    // fprintf(gnuplotPipe, "e\n");
    // pclose(gnuplotPipe);


    // InitWindow(WINDOW_W, WINDOW_H, "Diffusion FCS Simulation");
    // SetTargetFPS(60);

    // Camera3D cam = {
    //     .position   = { 30.0f, 20.0f, -20.0f },
    //     .target     = { 0.0f, 0.0f, 0.0f },
    //     .up         = { 0.0f, 1.0f, 0.0f },
    //     .fovy       = 45.0f,
    //     .projection = CAMERA_PERSPECTIVE,
    // };

    // while (!WindowShouldClose()) {
    //     step_sim(sim, DELTA_T);

    //     BeginDrawing();
    //         ClearBackground(BLACK);
    //         BeginMode3D(cam);
    //             // boundary box
    //             DrawCubeWires((Vector3){0,0,0}, BOUNDARY*2, BOUNDARY*2, BOUNDARY*2, DARKGRAY);
    //             // particles
    //             for (int i = 0; i < NUM_PARTICLES; i++) {
    //                 Vector3 pos = {
    //                     (float)coords[i]->x,
    //                     (float)coords[i]->y,
    //                     (float)coords[i]->z,
    //                 };
    //                 Color col = ColorFromHSV((float)i / NUM_PARTICLES * 360.0f, 0.9f, 1.0f);
    //                 DrawSphere(pos, SPHERE_RADIUS, col);
    //             }
    //         EndMode3D();
    //         DrawFPS(10, 10);
    //     EndDrawing();
    // }

    // CloseWindow();
    sim_destroy(sim);
    free_confocal_volume(volume);
    free_time_trace(trace);
    free_trace_autocorrelation(autocorr);
    return 0;
}