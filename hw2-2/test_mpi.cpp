#include <iostream>
#include <stdio.h>
#include <mpi.h>
void printf_status(MPI_Status *status) {
    std::cout << "count_lo " << status->count_lo << " count_hi_and cacelled " << status->count_hi_and_cancelled << " MPI_SOURCE: " << status->MPI_SOURCE << " MPI_TAG " << status->MPI_TAG << std::endl;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        char sendbuf[10] = "1fuckyou";
        MPI_Send(sendbuf, 10, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        char sendbuf2[10] = "11fuckyou";
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 2, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 3, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 4, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 5, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 6, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 7, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 8, MPI_COMM_WORLD);
        MPI_Send(sendbuf2, 10, MPI_CHAR, 1, 9, MPI_COMM_WORLD);
    }
    if (rank == 1) {
        char sendbuf[10] = "0fuckyou";
        MPI_Send(sendbuf, 10, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        char sendbuf2[10] = "00fuckyou";
        MPI_Send(sendbuf2, 10, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    char recvbuf[10];

    if (rank == 1) {
        MPI_Status status;
        printf("proc %d probe\n", rank);
        MPI_Probe(MPI_ANY_SOURCE, 9, MPI_COMM_WORLD, &status);
        printf_status(&status);
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf_status(&status);
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf_status(&status);
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf_status(&status);
    }

    if (rank == 1) {
        MPI_Recv(recvbuf, 10, MPI_CHAR, 0, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s\n", recvbuf);

    }
    // MPI_Recv(recvbuf, 10, MPI_CHAR, (rank + 1) % 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // printf("%s\n", recvbuf);

}