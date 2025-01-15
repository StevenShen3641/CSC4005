//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #2: Parallel Quick Sort with K-Way Merge using MPI
//

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>

#include <mpi.h>

#include "../utils.hpp"

#define MASTER 0
void seq_quickSort(std::vector<int> &vec, int low, int high);

int partition(std::vector<int> &vec, int low, int high)
{
    int pivot = vec[high];
    int i = low - 1;
    for (int j = low; j < high; j++)
    {
        if (vec[j] <= pivot)
        {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }
    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

/**
 * TODO: Implement parallel quick sort with MPI
 */
void quickSort(std::vector<int> &vec, int numtasks, int taskid,
               MPI_Status *status)
{
    /* Your codes here! */
    int n = vec.size();

    int chunk_size = n / numtasks;
    int remainder = n % numtasks;
    int local_size = chunk_size + (taskid < remainder ? 1 : 0);
    int start = taskid * chunk_size + std::min(taskid, remainder);
    int end = start + chunk_size + (taskid < remainder ? 1 : 0);
    std::vector<int> local_vec(vec.begin() + start, vec.begin() + end);

    seq_quickSort(local_vec, 0, local_size - 1);
    // std::sort(local_vec.begin(), local_vec.end());

    if (taskid == MASTER)
    {
        std::vector<std::vector<int>> sorted_subarrays(numtasks);
        sorted_subarrays[MASTER] = std::move(local_vec);

        for (int i = 1; i < numtasks; ++i)
        {
            int start = i * chunk_size + std::min(i, remainder);
            int end = start + chunk_size + (i < remainder ? 1 : 0);

            sorted_subarrays[i].resize(end - start);
            MPI_Recv(sorted_subarrays[i].data(), end - start, MPI_INT, i, 0,
                     MPI_COMM_WORLD, status);
        }

        using Element =
            std::tuple<int, int, int>; // (value, array index, position)
        std::priority_queue<Element, std::vector<Element>,
                            std::greater<Element>>
            min_heap;

        for (int i = 0; i < numtasks; ++i)
        {
            if (!sorted_subarrays[i].empty())
            {
                min_heap.emplace(sorted_subarrays[i][0], i, 0);
            }
        }

        int index = 0;
        while (!min_heap.empty())
        {
            Element elem = min_heap.top();
            min_heap.pop();
            int val = std::get<0>(elem);
            int array_idx = std::get<1>(elem);
            int pos = std::get<2>(elem);

            vec[index++] = val;

            if (pos + 1 < sorted_subarrays[array_idx].size())
            {
                min_heap.emplace(sorted_subarrays[array_idx][pos + 1],
                                 array_idx, pos + 1);
            }
        }

    }
    else
    {
        MPI_Send(local_vec.data(), local_vec.size(), MPI_INT, MASTER, 0,
                 MPI_COMM_WORLD);
    }
}

void seq_quickSort(std::vector<int> &vec, int low, int high)
{
    if (low < high)
    {
        int pivotIndex = partition(vec, low, high);
        seq_quickSort(vec, low, pivotIndex - 1);
        seq_quickSort(vec, pivotIndex + 1, high);
    }
}

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        throw std::invalid_argument("Invalid argument, should be: ./executable "
                                    "dist_type vector_size\n");
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

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

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
