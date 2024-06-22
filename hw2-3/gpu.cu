#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "common.h"
#include <cuda.h>
#include <vector>
#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;


__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

__device__ int get_bin_idx(double x, double y, int bin_heigh) {
    int bin_x = floor(x / bin_size);
    int bin_y = floor(y / bin_size);
    return bin_x * bin_heigh + bin_y;
}


__global__ void accumulate_bins_parts_cnt(particle_t* parts, int num_parts, int bin_height, int *bins_d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_parts) {
        particle_t* cur_part = &parts[tid];
        int bin_idx = get_bin_idx(cur_part->x, cur_part->y, bin_height);
        cur_part->bin_idx = bin_idx;
        atomicAdd(&bins_d[bin_idx], 1);
    }
}
// 对bins进行原地scan
void exclusive_scan_bins(int bins_len, int *scan_res, int *bins_d) {
    thrust::device_ptr<int> scan_res_vec(scan_res);
    thrust::device_ptr<int> bins_vec(bins_d);
    thrust::exclusive_scan(bins_vec, bins_vec + bins_len, scan_res_vec);
    CHECK(cudaMemcpy(bins_d, scan_res, sizeof(int) * bins_len, cudaMemcpyDeviceToDevice));
}

__global__ void distribute(particle_t* parts, int num_parts, int* scan_res, int *parts_idx_d) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_parts) {
        particle_t* cur_part = &parts[tid];
        parts_idx_d[atomicAdd(&scan_res[cur_part->bin_idx],1)] = tid;
    }
}
void distribute_parts_to_bin(particle_t* parts, int num_parts, int bin_height, int *bins_d, int *parts_idx_d) {
    int bins_len = bin_height * bin_height;
    // 每次accumulate之前都要将bins置为0
    CHECK(cudaMemset(bins_d, 0, bins_len * sizeof(int)));
    accumulate_bins_parts_cnt<<<blks, NUM_THREADS>>>(parts, num_parts, bin_height, bins_d);
    cudaDeviceSynchronize();
    int *scan_res;
    CHECK(cudaMalloc(&scan_res, sizeof(int) * bins_len));
    exclusive_scan_bins(bins_len, scan_res, bins_d);
    // 只改变了scan_res，没有改变真正的bins_d，因为之后要靠bins来访问每个bin对应的粒子
    distribute<<<blks, NUM_THREADS>>>(parts, num_parts, scan_res, parts_idx_d);
    cudaDeviceSynchronize();
    CHECK(cudaFree(scan_res));
}
// parts数组在gpu上
void init_simulation(particle_t* parts, int num_parts, double size, int **bins_d, int **parts_idx_d) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    int bin_height = (int)ceil(size / bin_size);
    int bin_width = bin_height;
    int bins_len = bin_height * bin_width;
    cudaMalloc(bins_d, sizeof(int) * bins_len);
    cudaMalloc(parts_idx_d, sizeof(int) * num_parts);
    cudaDeviceSynchronize();
    distribute_parts_to_bin(parts, num_parts, bin_height, *bins_d, *parts_idx_d);
}
__constant__  int next[9][2] = {
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

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}
__device__ void apply_force_from_bin(particle_t* parts, int bin_parts_start_idx, int bin_parts_end_idx, particle_t* cur_part, int *parts_idx_d) {
    for (int i = bin_parts_start_idx; i < bin_parts_end_idx; i++) {
        apply_force_gpu(*cur_part, parts[parts_idx_d[i]]);
    }
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int bin_height, int *bins_d, int *parts_idx_d) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int bins_len = bin_height * bin_height;
    particle_t* cur_part = &parts[tid];
    cur_part->ax = cur_part->ay = 0;
    double x = cur_part->x;
    double y = cur_part->y;
    // printf("bin_size %f\n", bin_size);
    int bin_x = floor(x / bin_size);
    int bin_y = floor(y / bin_size);
    for (int j = 0; j < 9; j++) {
        int nx = bin_x + next[j][0];
        int ny = bin_y + next[j][1];
        if (nx < 0 || nx >= bin_height || ny < 0 || ny >= bin_height) {
            continue;
        }
        int bin_idx = nx * bin_height + ny;
        // printf("%d\n", bin_idx);
        int bin_parts_start_idx = bins_d[bin_idx];
        int bin_parts_end_idx = (bin_idx + 1 == bins_len) ? num_parts : bins_d[bin_idx + 1];
        apply_force_from_bin(parts, bin_parts_start_idx, bin_parts_end_idx, cur_part, parts_idx_d);
    }

}

void simulate_one_step(particle_t* parts, int num_parts, double size, int *bins_d, int *parts_idx_d) {

    // parts live in GPU memory
    // Rewrite this function

    // Compute forces
    int bin_height = (int)ceil(size / bin_size);
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, bin_height, bins_d, parts_idx_d);
    cudaDeviceSynchronize();

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();
    distribute_parts_to_bin(parts, num_parts, bin_height, bins_d, parts_idx_d);
}
