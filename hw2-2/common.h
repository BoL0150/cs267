#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <cstdint>
#include <mpi.h>
#include <list>
// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005
#define bin_size cutoff
#define BIN_PARTS_NUM 3

// Particle Data Structure
typedef struct particle_t {
    bool valid;
    uint64_t id; // Particle ID
    double x;    // Position X
    double y;    // Position Y
    double vx;   // Velocity X
    double vy;   // Velocity Y
    double ax;   // Acceleration X
    double ay;   // Acceleration Y
} particle_t;

typedef struct bin {
  particle_t parts[BIN_PARTS_NUM];
} bin;

extern MPI_Datatype PARTICLE;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs);
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs);

#endif