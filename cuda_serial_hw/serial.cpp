#include <cassert>
#include <algorithm>
#include <stdio.h>
#include "common.h"
#include <cmath>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    // printf("part %d coef * dx: %f, coef * dy: %f\n", neighbor_idx, coef * dx, coef * dy);
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}
int get_bin_idx(double x, double y, int bin_heigh) {
    int bin_x = floor(x / bin_size);
    int bin_y = floor(y / bin_size);
    return bin_x * bin_heigh + bin_y;
}
int *bins;
int *parts_idx;
// 原地进行exclusive_scan
void exclusive_scan(int *array, int len) {
    long long prev_sum = 0;
    for (int i = 0; i < len; i++) {
        int temp = array[i];
        array[i] = prev_sum;
        prev_sum += temp;
    }
}
void distribute_parts_to_bin(particle_t* parts, int num_parts, int bin_height) {
    int bins_len = bin_height * bin_height;
    std::fill(bins, bins + bins_len, 0);
    for (int i = 0; i < num_parts; i++) {
        particle_t* cur_part = &parts[i];
        int bin_idx = get_bin_idx(cur_part->x, cur_part->y, bin_height);
        cur_part->bin_idx = bin_idx;
        // 计算每个bin中的粒子个数
        bins[bin_idx]++; 
    }
    exclusive_scan(bins, bins_len);
    int *temp_bins_idx = new int[bins_len];
    std::copy(bins, bins + bins_len, temp_bins_idx);
    for (int i = 0; i < num_parts; i++) {
        particle_t* cur_part = &parts[i];
        parts_idx[temp_bins_idx[cur_part->bin_idx]++] = i;
    }
}
void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    int bin_height = (int)ceil(size / bin_size);
    int bin_width = bin_height;
    int bins_len = bin_height * bin_width;
    bins = new int[bins_len];
    parts_idx = new int[num_parts];
    distribute_parts_to_bin(parts, num_parts, bin_height);
}

int next[9][2] = {
    {0, 1},
    {0, -1},
    {1, 0},
    {-1, 0},
    {-1, -1},
    {-1, 1},
    {1, 1},
    {1, -1},
    {0, 0}
};
void apply_force_from_bin(particle_t* parts, int bin_parts_start_idx, int bin_parts_end_idx, particle_t* cur_part) {
    for (int i = bin_parts_start_idx; i < bin_parts_end_idx; i++) {
        apply_force(*cur_part, parts[parts_idx[i]]);
    }
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // printf("bin_size %f\n", bin_size);
    int bin_height = (int)ceil(size / bin_size);
    int bins_len = bin_height * bin_height;
    for (int i = 0; i < num_parts; ++i) {
        particle_t* cur_part = &parts[i];
        cur_part->ax = cur_part->ay = 0;
        double x = cur_part->x;
        double y = cur_part->y;
        // printf("bin_size %f\n", bin_size);
        int bin_x = floor(x / bin_size);
        int bin_y = floor(y / bin_size);
        // printf("bin_size %f\n", bin_size);
        for (int j = 0; j < 9; j++) {
            int nx = bin_x + next[j][0];
            int ny = bin_y + next[j][1];
            if (nx < 0 || nx >= bin_height || ny < 0 || ny >= bin_height) {
                continue;
            }
            int bin_idx = nx * bin_height + ny;
            // printf("%d\n", bin_idx);
            int bin_parts_start_idx = bins[bin_idx];
            if (bin_idx + 1 == bins_len) assert(bin_parts_start_idx == num_parts);
            int bin_parts_end_idx = (bin_idx + 1 == bins_len) ? num_parts : bins[bin_idx + 1];
            apply_force_from_bin(parts, bin_parts_start_idx, bin_parts_end_idx, cur_part);
        }
    }
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
    distribute_parts_to_bin(parts, num_parts, bin_height);
    // printf("\n");
}
