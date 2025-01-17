//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2)
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
    // Considering Memory Locality and Avoiding Cache Missing
    // Hints:
    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication
    // const size_t CACHE_LINE_SIZE = 64; // load from sys
    // const size_t ELEMENT_SIZE = sizeof(int);
    const int tile_size = 64;

    // ensure spatial locality
    int* array1 = new int[M * K];
    int ind1 = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++) array1[ind1++] = matrix1[i][j];
    }

    int* array2 = new int[N * K];
    int ind2 = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++) array2[ind2++] = matrix2[j][i];
    }

    for (size_t i = 0; i < M; i += tile_size)
    {
        for (size_t j = 0; j < N; j += tile_size)
        {
            for (size_t ii = i; ii < std::min(i + tile_size, M); ++ii)
            {
                for (size_t jj = j; jj < std::min(j + tile_size, N); ++jj)
                {
                    int tmp =
                        0; // load from register, which is much more faster
                    for (size_t kk = 0; kk < K; ++kk)
                    {
                        tmp += array1[ii * K + kk] * array2[jj * K + kk];
                    }
                    result[ii][jj] = tmp; // load from a cache line
                }
            }
        }
    }

    delete[] array1;
    delete[] array2;

    return result;
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

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