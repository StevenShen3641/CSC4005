//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// OpenMp + SIMD + Reordering Matrix Multiplication
// scan

#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication
    // In addition to SIMD, Memory Locality and Cache Missing,
    // Further Applying OpenMp
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

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < M; i += tile_size)
    {
        for (size_t j = 0; j < N; j += tile_size)
        {
            for (size_t ii = i; ii < std::min(i + tile_size, M); ++ii)
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
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}