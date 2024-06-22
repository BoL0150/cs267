#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <stdio.h>
#include <memory>
#include "common.h"
#include <cmath>
// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor, int neighbor_idx) {
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
    // 粒子的坐标范围在0到x之间，闭区间
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

std::vector<std::vector<bin>> bins;
bin* get_bin(double x, double y) {
    int bin_x = floor(x / bin_size);
    int bin_y = floor(y / bin_size);
    return &bins[bin_x][bin_y];
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
void add_part_to_bin(bin* bin_ptr, particle_t* parts, int part_idx) {
    bin_ptr->part_idxs.push_back(part_idx);
    parts[part_idx].it = --bin_ptr->part_idxs.end();
}
void init_simulation(particle_t* parts, int num_parts, double size) {
	  // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    int bin_height = (int)ceil(size / bin_size);
    int bin_width = bin_height;
    bins = std::vector<std::vector<bin>>(bin_width, std::vector<bin>(bin_height));
    for (int i = 0; i < bin_width; i++) {
        for (int j = 0; j < bin_height; j++) {
            bins[i][j] = bin();
        }
    }
    for (int i = 0; i < num_parts; i++) {
        auto bin_ptr = get_bin(parts[i].x, parts[i].y);
        add_part_to_bin(bin_ptr, parts, i);
    }
}

void apply_force_to_bin(bin *neighbor_bin, int cur_part_idx, particle_t* parts) {
    for (int neighbor_part_idx : neighbor_bin->part_idxs) {
        apply_force(parts[cur_part_idx], parts[neighbor_part_idx], neighbor_part_idx);
    }
}
void compute_acce(particle_t* parts, int num_parts, double size) {
    // Compute Forces
	#pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        // printf("%d \n", omp_get_thread_num());
        parts[i].ax = parts[i].ay = 0;
        double x = parts[i].x;
        double y = parts[i].y;
        // printf("parts %d\n", i);
        // std::vector<std::shared_ptr<bin>> neighbor_bins = get_neighbor_bins(parts[i].x, parts[i].y, size);
        // 优化：函数融合，循环展开
        int bin_x = floor(x / bin_size);
        int bin_y = floor(y / bin_size);
        int bin_height = (int)ceil(size / bin_size);
        for (int j = 0; j < 9; j++) {
            int nx = bin_x + next[j][0];
            int ny = bin_y + next[j][1];
            if (nx < 0 || nx >= bin_height || ny < 0 || ny >= bin_height) {
                continue;
            }
            auto neighbor_bin = &bins[nx][ny];
            apply_force_to_bin(neighbor_bin, i, parts);
        }
        // printf("parts %d final ax:%f, ay:%f\n", i, parts[i].ax, parts[i].ay);
        // printf("\n");
    }
}
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute Forces
    compute_acce(parts, num_parts, size);
    // Move Particles
	#pragma omp single
    for (int i = 0; i < num_parts; ++i) {
        auto old_bin = get_bin(parts[i].x, parts[i].y);
        move(parts[i], size);
        // printf("parts %d position: x %f,y %f\n", i, parts[i].x,parts[i].y);
        auto new_bin = get_bin(parts[i].x, parts[i].y);
        if (old_bin == new_bin) continue;
        old_bin->part_idxs.erase(parts[i].it);
        add_part_to_bin(new_bin, parts, i);
    }
    // printf("\n");
}