#include <cassert>
#include "common.h"
#include <algorithm>
#include <stdio.h>
#include <memory>
#include <cmath>
#include <vector>
std::vector<particle_t*> local_parts;
double block_x1, block_x2, block_y1, block_y2;
double ghost_block_x1, ghost_block_x2, ghost_block_y1, ghost_block_y2;
std::vector<std::vector<bin>> local_bins;
double local_size;
int bin_height;
int rank_x;
int rank_y;
int x_proc_nums;
int total_proc_nums;
bool check_local_part(particle_t* part) {
    if (block_x1 <= part->x && part->x < block_x2 && block_y1 <= part->y && part->y < block_y2) {
        return true;
    }
    return false;
}
// 获取的是加上ghost particle之后的bin的坐标
int get_local_bin_x(double global_x) {
    assert(global_x >= block_x1 && global_x <= block_x2);
    double local_offset_x = global_x - block_x1;
    int bin_x = floor(local_offset_x / bin_size);
    return bin_x + 1;
}

int get_local_bin_y(double global_y) {
    assert(global_y >= block_y1 && global_y <= block_y2);
    double local_offset_y = global_y - block_y1;
    int bin_y = floor(local_offset_y / bin_size);
    return bin_y + 1;
}
bin* get_local_bin(double x, double y) {
    int bin_x = get_local_bin_x(x);
    int bin_y = get_local_bin_y(y);
    assert(bin_x >= 1 && bin_x <= bin_height - 2 && bin_y >= 1 && bin_y <= bin_height - 2);
    return &local_bins[bin_x][bin_y];
}
bin* get_local_and_ghost_bin(double x, double y) {
    assert(x >= ghost_block_x1 && x <= ghost_block_x2 && y >= ghost_block_y1 && y <= ghost_block_y2);
    if (x >= block_x1 && x <= block_x2 && y >= block_y1 && y <= block_y2) {
        return get_local_bin(x, y);
    }
    int bin_x = floor((x - ghost_block_x1) / bin_size);
    int bin_y = floor((y - ghost_block_y1) / bin_size);
    assert((bin_x == 0 || bin_x == bin_height - 1) && (bin_y == 0 || bin_y == bin_height - 1));
    bin* ghost_bin = &local_bins[bin_x][bin_y];
    return ghost_bin;
}
particle_t* copy_part_to_local_bin(bin* bin_ptr, particle_t *part) {
    particle_t* result = NULL;
    for (int i = 0; i < 3; i++) {
        if (bin_ptr->parts[i].valid == true) continue;
        bin_ptr->parts[i] = *part;
        result = &bin_ptr->parts[i];
        bin_ptr->parts[i].valid = true;
        break;
    }
    assert(result != NULL);
    return result;
}
// 将全局的part数组中分给当前进程的parts分配并复制到当前进程的bin中，然后再将bin中存储的parts对象的指针保存到local_parts中
// 所以在初始化时，每个粒子在整个系统中都有两个一样的副本，一个在全局的parts数组中，一个在被分配的进程的bin中，
// 每个进程的local_parts数组中保存的指针指向的是进程本地bin中的particle实例
// 模拟开始之后，每一步更新的是进程本地的bin中的粒子，而不是全局parts数组中的粒子
void distribute_to_local_bins(particle_t* parts, int num_parts) {
    local_bins = std::vector<std::vector<bin>>(bin_height, std::vector<bin>(bin_height));
    for (int i = 0; i < bin_height; i++) {
        for (int j = 0; j < bin_height; j++) {
            local_bins[i][j] = bin();
            for (int k = 0; k < BIN_PARTS_NUM; k++) {
                local_bins[i][j].parts[k].valid = false;
            }
        }
    }
    for (int i = 0; i < num_parts; i++) {
        particle_t* cur_part = &parts[i];
        if (check_local_part(cur_part)) {
            auto bin_ptr = get_local_bin(cur_part->x, cur_part->y);
            particle_t* res = copy_part_to_local_bin(bin_ptr, cur_part);
            local_parts.push_back(res);
        }
    }
}
MPI_Datatype BIN;
// Put any static global variables here that you will use throughout the simulation.
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    // 假设是正方形
    x_proc_nums = floor(sqrt(num_procs));
    int y_proc_nums = x_proc_nums; 
    total_proc_nums = x_proc_nums * y_proc_nums;
    local_size = size / x_proc_nums;
    // 多两行和两列，ghost particles，便于从其他进程获取数据
    bin_height = (int)ceil(local_size / bin_size) + 2;
    rank_x = rank / y_proc_nums;
    rank_y = rank % y_proc_nums;
    block_x1 = rank_x * local_size;
    block_x2 = (rank_x + 1) * local_size;
    block_y1 = rank_y * local_size;
    block_y2 = (rank_y + 1) * local_size;
    ghost_block_x1 = block_x1 - bin_size;
    ghost_block_y1 = block_y1 - bin_size;
    ghost_block_x2 = block_x2 + bin_size;
    ghost_block_y2 = block_y2 + bin_size;
    distribute_to_local_bins(parts, num_parts);

    const int nitems = 1;
    int blocklengths = BIN_PARTS_NUM;
    MPI_Datatype types = PARTICLE;
    MPI_Aint offset = offsetof(bin, parts);
    MPI_Type_create_struct(nitems, &blocklengths, &offset, &types, &BIN);
    MPI_Type_commit(&BIN);
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
void apply_force_to_bin(bin *neighbor_bin, particle_t* cur_part) {
    for (int i = 0; i < BIN_PARTS_NUM; i++) {
        if (!neighbor_bin->parts[i].valid) continue;
        apply_force(*cur_part, neighbor_bin->parts[i]);
    }
}
void compute_acce(void) {
    // Compute Forces
    for (int i = 0; i < local_parts.size(); ++i) {
        particle_t* cur_part = local_parts[i];
        cur_part->ax = cur_part->ay = 0;
        double x = cur_part->x;
        double y = cur_part->y;
        // printf("parts %d\n", i);
        int bin_x = get_local_bin_x(x);
        int bin_y = get_local_bin_y(y);
        for (int j = 0; j < 9; j++) {
            int nx = bin_x + next[j][0];
            int ny = bin_y + next[j][1];
            assert(nx >= 0 && nx < bin_height && ny >= 0 && ny < bin_height);
            auto neighbor_bin = &local_bins[nx][ny];
            apply_force_to_bin(neighbor_bin, cur_part);
        }
        // printf("parts %d final ax:%f, ay:%f\n", i, parts[i].ax, parts[i].ay);
        // printf("\n");
    }
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
int get_1D_rank(int x, int y) {
    return x * x_proc_nums + y;
}
void mpi_sendrecv_warpper(bin *sendbuf, int dest, bin *recvbuf, int source, bool update) {
    if (!update) {
        MPI_Sendrecv(sendbuf, 1, BIN, dest, 0, recvbuf, 1, BIN, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return;
    } 
    // 如果是update的话就要把之前发出去的bin收回来，更新自己本地的bin，然后也要把之前接收的bin发回去
    bin temp;
    MPI_Sendrecv(recvbuf, 1, BIN, dest, 0, &temp, 1, BIN, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < BIN_PARTS_NUM; i++) {
        if (temp.parts[i].valid) {
            for (int j = 0; j < BIN_PARTS_NUM; j++) {
                if (!sendbuf->parts[j].valid) {
                    sendbuf->parts[j] = temp.parts[i];
                    sendbuf->parts[j].valid = true;
                    break;
                } else {
                    assert(temp.parts[i].id != sendbuf->parts[j].id);
                }
            }
        }
    }
}
void communicate(int rank, bool update) {
    for (int i = 0; i < bin_height; i++) {
        for (int j = 0; j < bin_height; j++) {
            if (i > 0 && i < bin_height - 1 && j > 0 && j < bin_height - 1) continue;
            if (i == 0 && rank_x == 0) continue;
            if (i == bin_height - 1 && rank_x == x_proc_nums - 1) continue;
            if (j == 0 && rank_y == 0) continue;
            if (j == bin_height - 1 && rank_y == x_proc_nums - 1) continue;
            if (i == 0 && j == 0) {
                int dest = get_1D_rank(rank_x - 1, rank_y - 1);
                mpi_sendrecv_warpper(&local_bins[1][1], dest, &local_bins[i][j], rank, update);
            } else if (i == 0 && j == bin_height - 1) {
                int dest = get_1D_rank(rank_x - 1, rank_y + 1);
                mpi_sendrecv_warpper(&local_bins[1][bin_height - 2], dest, &local_bins[i][j], rank, update);
            } else if (i == bin_height - 1 && j == 0) {
                int dest = get_1D_rank(rank_x + 1, rank_y - 1);
                mpi_sendrecv_warpper(&local_bins[bin_height - 2][1], dest, &local_bins[i][j], rank, update);
            } else if (i == bin_height - 1 && j == bin_height - 1) {
                int dest = get_1D_rank(rank_x + 1, rank_y + 1);
                mpi_sendrecv_warpper(&local_bins[bin_height - 2][bin_height - 2], dest, &local_bins[i][j], rank, update);
            } else if (i == 0) {
                int dest = get_1D_rank(rank_x - 1, rank_y);
                mpi_sendrecv_warpper(&local_bins[1][j], dest, &local_bins[i][j], rank, update);
            } else if (j == 0) {
                int dest = get_1D_rank(rank_x, rank_y - 1);
                mpi_sendrecv_warpper(&local_bins[i][1], dest, &local_bins[i][j], rank, update);
            } else if (i == bin_height - 1) {
                int dest = get_1D_rank(rank_x + 1, rank_y);
                mpi_sendrecv_warpper(&local_bins[bin_height - 2][j], dest, &local_bins[i][j], rank, update);
            } else {
                int dest = get_1D_rank(rank_x, rank_y + 1);
                mpi_sendrecv_warpper(&local_bins[i][bin_height - 2], dest, &local_bins[i][j], rank, update);
            }
        }
    }

}
void clear_ghost_bin() {
    for (int i = 0; i < bin_height; i++) {
        for (int j = 0; j < bin_height; j++) {
            if (i == 0 || j == 0 || i == bin_height - 1 || j == bin_height - 1) {
                for (int k = 0; k < BIN_PARTS_NUM; k++) {
                    local_bins[i][j].parts[k].valid = false;
                }
            }
        }
    }
}
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // 让多余的proc空闲
    if (rank >= total_proc_nums) return;
    // 和周围的进程通信，获取ghost bin，这些ghost bin只存在于local_bins数组中，他们的particle不需要加入local_parts中
    communicate(rank, false);
    // Compute Forces
    compute_acce();
    // 在move之前把ghost bin中的粒子清空，从而方便后面将ghost bin发送回去之后其他进程更新自己的bin
    clear_ghost_bin();
    // 移动粒子，只负责移动本地的粒子，有可能会移动到ghost particle中，ghost particle中的粒子只会多不会少
    for (int i = 0; i < local_parts.size(); ++i) {
        particle_t *cur_part = local_parts[i];
        bin* old_bin = get_local_bin(cur_part->x, cur_part->y);
        move(*cur_part, size);
        // printf("parts %d position: x %f,y %f\n", i, parts[i].x,parts[i].y);
        bin* new_bin = get_local_and_ghost_bin(cur_part->x, cur_part->y);
        if (old_bin == new_bin) continue;
        // 目前part保存在old_bin中，所以是将old_bin中的valid变成false
        cur_part->valid = false;
        // 将local_parts中的旧的指针替换成新的bin中的part的指针
        local_parts[i] = copy_part_to_local_bin(new_bin, cur_part);
    }
    // 在move之后把之前发出去的bin收回来，更新自己本地的bin，然后也要把之前接收的bin发回去
    communicate(rank, true);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    int send_parts_num = local_parts.size();
    particle_t send_buf[send_parts_num];
    for (int i = 0; i < send_parts_num; i++) {
        send_buf[i] = *local_parts[i];
    }
    // 根进程从每个进程接收数据，按照rank排列在revc_buf中，每个进程发送数据个数是send_parts_num，根进程从每个进程接收的数据个数也是send_parts_num
    MPI_Gather(send_buf, send_parts_num, PARTICLE, parts, send_parts_num, PARTICLE, 0, MPI_COMM_WORLD);
    // 按照id升序排序
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](particle_t &a, particle_t &b) {
            return a.id < b.id;
        });
    }
}