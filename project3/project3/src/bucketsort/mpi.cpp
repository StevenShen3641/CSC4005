//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #1: Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int>& bucket)
{
    /* You may print out the data size in each bucket here to see how severe the
     * load imbalance is */
    // if (bucket.size() >0) std::cout<<bucket.size()<<std::endl;
    for (int i = 1; i < bucket.size(); ++i)
    {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key)
        {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

/**
 * TODO: Parallel Bucket Sort with MPI
 * @param vec: input vector for sorting
 * @param num_buckets: number of buckets
 * @param numtasks: number of processes for sorting
 * @param taskid: the rank of the current process
 * @param status: MPI_Status for message passing
 */
void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks,
                int taskid, MPI_Status* status)
{
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = min_val + large_bucket_num * large_bucket_size;

    int buckets_per_process = num_buckets / numtasks;
    int extra_buckets = num_buckets % numtasks;
    int start_bucket =
        taskid * buckets_per_process + std::min(taskid, extra_buckets);
    int end_bucket =
        start_bucket + buckets_per_process + (taskid < extra_buckets ? 1 : 0);
    int local_num_buckets = end_bucket - start_bucket;

    std::vector<std::vector<int>> buckets(local_num_buckets);
    for (std::vector<int>& bucket : buckets)
    {
        bucket.reserve(large_bucket_size);
    }

    int local_min, local_max;

    if (start_bucket < large_bucket_num)
    {
        local_min = min_val + start_bucket * large_bucket_size;
        local_max = min_val + end_bucket * small_bucket_size +
                    std::min(end_bucket, large_bucket_num);
    }
    else
    {
        local_min =
            boundary + (start_bucket - large_bucket_num) * small_bucket_size;
        local_max =
            boundary + (end_bucket - large_bucket_num) * small_bucket_size;
    }

    for (int num : vec)
    {
        if (num >= local_min && num < local_max)
        {
            int index;
            if (num < boundary)
            {
                index = (num - min_val) / large_bucket_size;
            }
            else
            {
                index = large_bucket_num + (num - boundary) / small_bucket_size;
            }
            buckets[index - start_bucket].push_back(num);
        }
    }

    for (std::vector<int>& bucket : buckets)
    {
        insertionSort(bucket);
    }

    if (taskid == MASTER)
    {
        int index = 0;
        for (const std::vector<int>& bucket : buckets)
        {
            for (int num : bucket)
            {
                vec[index++] = num;
            }
        }

        for (int i = 1; i < numtasks; i++)
        {
            int num_elements;
            MPI_Recv(&num_elements, 1, MPI_INT, i, 1, MPI_COMM_WORLD, status);

            std::vector<int> recv_buf(num_elements);
            MPI_Recv(recv_buf.data(), num_elements, MPI_INT, i, 2,
                     MPI_COMM_WORLD, status);
            for (int num : recv_buf)
            {
                vec[index++] = num;
            }
        }
    }
    else
    {
        int num_elements = 0;
        for (const std::vector<int>& bucket : buckets)
        {
            num_elements += bucket.size();
        }

        std::vector<int> send_buf;
        for (const std::vector<int>& bucket : buckets)
        {
            send_buf.insert(send_buf.end(), bucket.begin(), bucket.end());
        }

        MPI_Send(&num_elements, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(send_buf.data(), num_elements, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument("Invalid argument, should be: ./executable "
                                    "dist_type vector_size bucket_num\n");
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

    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    const int bucket_num = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
