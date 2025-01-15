//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h> // MPI Header
#include <omp.h>
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2,
                           int taskid, int numtasks)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    Matrix result(M, N);

    const int tile_size = 64;

    // ensure spatial locality
    int* array1 = (int*)_mm_malloc(M * K * sizeof(int), 32);
    int ind1 = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++) array1[ind1++] = matrix1[i][j];
    }

    int* array2 = (int*)_mm_malloc(N * K * sizeof(int), 32);
    int ind2 = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++) array2[ind2++] = matrix2[j][i];
    }

    // Divide the matrix rows among processes
    size_t rows_per_process = (M + numtasks - 1) / numtasks;
    size_t start_row = taskid * rows_per_process;
    size_t end_row = std::min(start_row + rows_per_process, M);

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = start_row; i < end_row; i += tile_size)
    {
        for (size_t j = 0; j < N; j += tile_size)
        {
            for (size_t ii = i; ii < std::min(i + tile_size, end_row); ++ii)
            {
                for (size_t jj = j; jj < std::min(j + tile_size, N); ++jj)
                {
                    __m256i tmp = _mm256_setzero_si256();
                    size_t kk;

                    for (kk = 0; kk + 16 <= K; kk += 16)
                    {
                        __m256i a1 =
                            _mm256_load_si256((__m256i*)&array1[ii * K + kk]);
                        __m256i a2 = _mm256_load_si256(
                            (__m256i*)&array1[ii * K + kk + 8]);

                        __m256i b1 =
                            _mm256_load_si256((__m256i*)&array2[jj * K + kk]);
                        __m256i b2 = _mm256_load_si256(
                            (__m256i*)&array2[jj * K + kk + 8]);

                        __m256i mul1 = _mm256_mullo_epi32(a1, b1);
                        __m256i mul2 = _mm256_mullo_epi32(a2, b2);

                        tmp = _mm256_add_epi32(tmp, mul1);
                        tmp = _mm256_add_epi32(tmp, mul2);
                    }

                    __m128i tmp_low = _mm256_castsi256_si128(tmp);
                    __m128i tmp_high = _mm256_extracti128_si256(tmp, 1);
                    __m128i sum_128 = _mm_add_epi32(tmp_low, tmp_high);
                    sum_128 = _mm_hadd_epi32(sum_128, sum_128);
                    sum_128 = _mm_hadd_epi32(sum_128, sum_128);

                    int sum = _mm_cvtsi128_si32(sum_128);

                    for (; kk < K; ++kk)
                    {
                        sum += array1[ii * K + kk] * array2[jj * K + kk];
                    }

                    result[ii][jj] = sum;
                }
            }
        }
    }

    _mm_free(array1);
    _mm_free(array2);

    return result;
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 5)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER)
    {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, taskid, numtasks);

        // Your Code Here for Synchronization!
        for (int source = 1; source < numtasks; source++)
        {
            size_t rows_per_process =
                (matrix1.getRows() + numtasks - 1) / numtasks;

            size_t source_start_row = source * rows_per_process;
            size_t source_rows = std::min(rows_per_process,
                                          matrix1.getRows() - source_start_row);
            for (size_t i = 0; i < source_rows; i++)
            {
                MPI_Recv(&result[source_start_row + i][0], matrix2.getCols(),
                         MPI_INT, source, 0, MPI_COMM_WORLD, &status);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    }
    else
    {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, taskid, numtasks);

        // Your Code Here for Synchronization!
        size_t rows_per_process = (matrix1.getRows() + numtasks - 1) / numtasks;
        size_t start_row = taskid * rows_per_process;
        size_t rows_to_send =
            std::min(rows_per_process, matrix1.getRows() - start_row);

        for (size_t i = 0; i < rows_to_send; i++)
        {
            MPI_Send(&result[start_row + i][0], matrix2.getCols(), MPI_INT,
                     MASTER, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}