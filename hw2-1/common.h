#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#include <vector>
#include <list>
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005
#define bin_size ((4.0 * cutoff) / 4)

// Particle Data Structure
typedef struct particle_t {
    double x;  // Position X
    double y;  // Position Y
    double vx; // Velocity X
    double vy; // Velocity Y
    double ax; // Acceleration X
    double ay; // Acceleration Y
    std::list<int>::iterator it;
} particle_t;

typedef struct bin {
  std::list<int> part_idxs;
}bin;

// Simulation routine
void init_simulation(particle_t* parts, int num_parts, double size);
void simulate_one_step(particle_t* parts, int num_parts, double size);

#endif
