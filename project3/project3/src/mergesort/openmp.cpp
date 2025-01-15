//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #4: Parallel Merge Sort with OpenMP
//

#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
#include "../utils.hpp"

/**
 * TODO: Implement parallel merge algorithm
 */
void merge(std::vector<int>& vec, int l, int mid, int r, int thread_num,
           int num_tasks)
{
    if (r - l <= 100)
    {
        std::vector<int> temp(r - l + 1);
        int i = l, j = mid + 1, index = 0;

        while (i <= mid && j <= r)
        {
            if (vec[i] <= vec[j])
            {
                temp[index++] = vec[i++];
            }
            else
            {
                temp[index++] = vec[j++];
            }
        }

        while (i <= mid) temp[index++] = vec[i++];
        while (j <= r) temp[index++] = vec[j++];

        for (i = l; i <= r; i++)
        {
            vec[i] = temp[i - l];
        }
    }
    else if (num_tasks < thread_num)
    {
        int merge_size = (mid - l + 1 + r - mid) / 2;
        int lp = l - 1, rp = mid;
        while (true)
        {
            if (merge_size == 1)
            {
                if (vec[lp + 1] <= vec[rp + 1])
                {
                    lp++;
                }
                else
                {
                    rp++;
                }
                break;
            }
            int ll = std::min(mid, lp + (merge_size / 2));
            int rl = std::min(r, rp + (merge_size / 2));
            int diff;
            if (vec[ll] < vec[rl])
            {
                diff = ll - lp;
                merge_size -= diff;
                lp = ll;

                if (lp == mid)
                {
                    rp += merge_size;
                    break;
                }
            }
            else
            {
                diff = rl - rp;
                merge_size -= diff;
                rp = rl;

                if (rp == r)
                {
                    lp += merge_size;
                    break;
                }
            }
        }

        auto temp_l = vec.begin() + lp + 1;
        auto temp_m = vec.begin() + mid + 1;
        auto temp_r = vec.begin() + rp + 1;

        std::vector<int> temp(temp_l, temp_m);
        std::move(temp_m, temp_r, temp_l);
        std::move(temp.begin(), temp.end(), temp_r - temp.size());

#pragma omp task shared(vec)
        merge(vec, l, lp, lp + (rp - mid), thread_num, num_tasks * 2);

#pragma omp task shared(vec)
        merge(vec, lp + (rp - mid) + 1, rp, r, thread_num, num_tasks * 2);

#pragma omp taskwait
    }
    else
    {
        int n1 = mid - l + 1;
        int n2 = r - mid;

        std::vector<int> L(n1), R(n2);

        for (int i = 0; i < n1; i++)
        {
            L[i] = vec[l + i];
        }
        for (int i = 0; i < n2; i++)
        {
            R[i] = vec[mid + 1 + i];
        }

        int i = 0;
        int j = 0;
        int k = l;
        while (i < n1 && j < n2)
        {
            if (L[i] <= R[j])
            {
                vec[k] = L[i];
                i++;
            }
            else
            {
                vec[k] = R[j];
                j++;
            }
            k++;
        }

        while (i < n1)
        {
            vec[k] = L[i];
            i++;
            k++;
        }

        while (j < n2)
        {
            vec[k] = R[j];
            j++;
            k++;
        }
    }
}

/**
 * TODO: Implement parallel merge sort by dynamic threads creation
 */
void mergeSort(std::vector<int>& vec, int l, int r, int thread_num,
               int num_tasks)
{
    /* Your codes here! */
    if (num_tasks < thread_num)
    {
        if (l < r)
        {
            if (r - l >= 32)
            {
                int mid = (l + r) / 2;
#pragma omp taskgroup
                {
#pragma omp task shared(vec) untied if (r - l >= (1 << 14))
                    mergeSort(vec, l, mid, thread_num, num_tasks << 1);
#pragma omp task shared(vec) untied if (r - l >= (1 << 14))
                    mergeSort(vec, mid + 1, r, thread_num, num_tasks << 1);
#pragma omp taskyield
                }
                merge(vec, l, mid, r, thread_num, num_tasks);
            }
            else
            {
                std::sort(vec.begin() + l, vec.begin() + r + 1);
            }
        }
    }
    else
    {
        std::sort(vec.begin() + l, vec.begin() + r + 1);
    }
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument("Invalid argument, should be: ./executable "
                                    "dist_type threads_num vector_size\n");
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int thread_num = atoi(argv[2]);
    const int size = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    // std::for_each(vec.begin(), vec.end(),
    //               [](int value) { std::cout << value << " "; });
    // std::cout << std::endl;

#pragma omp parallel
    {
#pragma omp single
        mergeSort(vec, 0, size - 1, thread_num, 1);
    }

    // std::for_each(vec.begin(), vec.end(),
    //               [](int value) { std::cout << value << " "; });
    // std::cout << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
    return 0;
}
