#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
// #include <mpi.h>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << "\n";
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << "\n";
    }

    fsave << std::endl;
}

// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }

    for (int i = 0; i < num_parts; ++i) {
        parts[i].id = i + 1;
        // printf("part id:%d %f %f\n",parts[i].id, parts[i].x, parts[i].y);
    }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

MPI_Datatype PARTICLE;

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set number of particles" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);

    // Init MPI
    int num_procs, rank;
    // MPI_Init后的代码所有的进程都要执行
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Create MPI Particle Type
    // 结构体PARTICLE中的成员数量
    const int nitems = 8;
    // 每个成员的数量
    int blocklengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    // MPI类型中的每个成员的具体类型
    MPI_Datatype types[8] = {MPI_C_BOOL, MPI_UINT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                             MPI_DOUBLE,   MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[8];
    // 每个成员在PARTICLE结构体中的偏移量
    offsets[0] = offsetof(particle_t, valid);
    offsets[1] = offsetof(particle_t, id);
    offsets[2] = offsetof(particle_t, x);
    offsets[3] = offsetof(particle_t, y);
    offsets[4] = offsetof(particle_t, vx);
    offsets[5] = offsetof(particle_t, vy);
    offsets[6] = offsetof(particle_t, ax);
    offsets[7] = offsetof(particle_t, ay);
    // MPI不能发送和接收自定义的类对象，所以需要使用MPI_Type_create_struct来创建一个类型
    // 之后就可以使用PARTICLE数据类型来接收和发送particle_t结构体的实例
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    // Initialize Particles
    int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    double size = sqrt(density * num_parts);

    particle_t* parts = new particle_t[num_parts];

    if (rank == 0) {
        init_particles(parts, num_parts, size, part_seed);
    }
    // 将粒子数据从根进程广播到所有其他进程，确保所有进程在模拟开始时都使用相同的初始数据
    // 其他进程不需要显式调用任何MPI函数来接收数据，自动将根进程parts数组的内容同步到所有的进程的parts数组
    // 每个进程都有一份完整的parts数组的副本
    MPI_Bcast(parts, num_parts, PARTICLE, 0, MPI_COMM_WORLD);

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();

    // 此时所有进程中的parts数组的内容是相同的，每个进程将parts数组的一部分保存在自己的地址空间中，然后
    // simulate_one_step中对这部分粒子进行操作
    init_simulation(parts, num_parts, size, rank, num_procs);

    for (int step = 0; step < nsteps; ++step) {
        // 每个进程都有一份完整的parts数组，但是其中只有自己进程的那部分parts是正确的数据，
        // parts数组中其他部分都是旧的数据，不能使用
        simulate_one_step(parts, num_parts, size, rank, num_procs);

        // Save state if necessary
        if (fsave.good() && (step % savefreq) == 0) {
            // 每一次simulate_one_step之后，gather负责将所有进程的本地的parts更新到根进程的parts数组中，
            // 然后将全局的parts数组保存下来。
            gather_for_save(parts, num_parts, size, rank, num_procs);
            if (rank == 0) {
                save(fsave, parts, num_parts, size);
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    if (rank == 0) {
        std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts
                  << " particles.\n";
    }
    if (fsave) {
        fsave.close();
    }
    delete[] parts;
    MPI_Finalize();
}
