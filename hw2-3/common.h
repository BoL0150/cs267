#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__
#include <list>
#include <stdio.h>
#define CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005
#define bin_size cutoff

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
    int bin_idx;
} particle_t;


// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size, int **bins_d, int **parts_idx_d);
void simulate_one_step(particle_t* parts, int num_parts, double size, int *bins_d, int *parts_idx_d);

#endif
