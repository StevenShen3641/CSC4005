## Project 3

#### Introduction

In this project, we are asked to implement the Process-Level Parallel Bucket Sort, Process-Level Parallel Quick Sort with K-Way Merge, Process-Level Parallel Sorting with Regular Sampling and Dynamic Thread-Level Parallel Merge Sort.

#### Compilation and Execution

To compile the program, please do the following steps:

```bash
cd project3
mkdir build && cd build
cmake ..
make -j4
```

After compilation, in order to batch process the project in order to get the execution time, you can simply `sbatch` at the project root directory:

```bash
cd /path/to/project3
sbatch ./src/sbatch-uniform.sh
sbatch ./src/sbatch-normal.sh
```

The result is stored in `Project3-uniform-results.txt` and `Project3-uniform-results.txt`. You can use `vim` to open it.

In order to get perf result, you can simply run the following commands:

```bash
#!/bin/bash

mkdir -p ./perf_results

DATA_SIZE=100000000
BUCKET_SIZE=1000000


for processes in 1 4 16 32; do
	# bucket
    mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/b_uniform_${processes}.data ./build/src/bucketsort/bucketsort_mpi uniform $DATA_SIZE $BUCKET_SIZE

	# quick
	mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/q_uniform_${processes}.data ./build/src/quicksort/quicksort_mpi uniform $DATA_SIZE
	mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/q_normal_${processes}.data ./build/src/quicksort/quicksort_mpi normal $DATA_SIZE

	# PSRS
	mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/psrs_uniform_${processes}.data ./build/src/psrs/psrs_mpi uniform $DATA_SIZE
	mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/psrs_normal_${processes}.data ./build/src/psrs/psrs_mpi normal $DATA_SIZE

	# merge
	srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/merge_uniform_${processes}.data ./build/src/mergesort/mergesort_openmp uniform $processes $DATA_SIZE
	srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/merge_normal_${processes}.data ./build/src/mergesort/mergesort_openmp normal $processes $DATA_SIZE

done
```

After that, you can use `perf report` to view the results under the `perf_results` folder. Note that for MPI, since `perf stat` is used, you can use `vim` to see the results.

#### Parallel Principle and Optimization

1. For bucket sort, we divided the bucket into several parts, send each part to a process, throw numbers into the buckets and do the sort locally. After that, we finally use concatenation to connect all values. The optimization I do here is to first compute the upper and lower bound of a process, so there is no need for all values to calculate their bucket index in each process.
2. For quick sort, we divide the data into multiple blocks and assign each block to an MPI process, which sorts it using a sequential quick-sort. Once all processes have completed sorting their respective blocks, a sequential merge is performed to combine the sorted blocks into the final ordered list. The optimization I do here is to use MPI_Bcast and use min heap for the final merge part.
3. For PSRS, it has four phases. In Phase One, each processor sorts a local block of data using quicksort and selects a representative sample of p items from their sorted block. These samples, one from each processor, represent the data distribution. In Phase Two, one processor gathers and sorts these samples, selecting p-1 pivots to partition the data. Each processor then forms p partitions based on the pivots. In Phase Three, processors exchange partitions, where each processor keeps one partition and sends the others to the respective processors. Finally, in Phase Four, each processor merges its partitions into a single sorted list, creating the final sorted array. The optimization I do here is to use MPI_Alltoall and MPI_Gather to centralize the message sending process.

4. For merge sort, here we do median binary search in the merge function as well as parallelize merge. We can also use `taskgroup` as introduced in [Advanced Programming with OpenMP (Quick Sort as one Example)](https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf). `#pragma omp task shared(vec) untied if (r - l >= (1 << 14))`. The intrinsic logic is we can create a new task automatically and only if the amount of work is sufficient. Then we just simply merge two arrays and the workload could dynamically be distributed by different tasks.

#### Result

The result is shown below:

For uniform:

| Workers | std::sort | BucketSort(MPI) | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) |
| ------- | --------- | --------------- | -------------- | ---------- | ------------------ |
| 1       | 11777     | 10875           | 13860          | 15325      | 10895              |
| 4       | N/A       | 3836            | 9187           | 5851       | 6750               |
| 16      | N/A       | 1711            | 8799           | 1827       | 2138               |
| 32      | N/A       | 1548            | 9133           | 1073       | 1394               |

The speedup factors are shown as followed, using $$S(p) = \frac{t_s}{t_p}$$ and Naive performance (11777 ms) as a baseline. 

| Workers | std::sort | BucketSort(MPI) | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) |
| ------- | --------- | --------------- | -------------- | ---------- | ------------------ |
| 1       | 1         | 1.08            | 0.85           | 0.77       | 1.08               |
| 4       | N/A       | 3.07            | 1.28           | 2.01       | 1.74               |
| 16      | N/A       | 6.88            | 1.34           | 6.45       | 5.51               |
| 32      | N/A       | 7.61            | 1.29           | 10.98      | 8.45               |

For normal:

| Workers | std::sort | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) |
| ------- | --------- | -------------- | ---------- | ------------------ |
| 1       | 11740     | 12958          | 15870      | 11633              |
| 4       | N/A       | 9230           | 5793       | 6066               |
| 16      | N/A       | 9105           | 2091       | 2016               |
| 32      | N/A       | 9219           | 1150       | 1456               |

The speedup factors are shown as followed, using $$S(p) = \frac{t_s}{t_p}$$ and Naive performance (11740 ms) as a baseline. 

| Workers | std::sort | QuickSort(MPI) | PSRS (MPI) | MergeSort (OpenMP) |
| ------- | --------- | -------------- | ---------- | ------------------ |
| 1       | 1         | 0.91           | 0.74       | 1.01               |
| 4       | N/A       | 1.28           | 2.03       | 1.94               |
| 16      | N/A       | 1.29           | 5.63       | 5.82               |
| 32      | N/A       | 1.28           | 10.24      | 8.06               |

We can see that PSRS has the fastest speed. With appropriately distribution of different values, the time can reduce to 1/10. What is more, uniform data seems to have a better performance on bucket sort, but the rest three are all have similar performances.

#### Perf Result

Since the perf results are very large in size, we only show processes = 4 and uniform distribution here.

Bucket Sort:

![image-20241119210730508](C:\Users\Chihao Shen\AppData\Roaming\Typora\typora-user-images\image-20241119210730508.png)

![image-20241119211005310](C:\Users\Chihao Shen\AppData\Roaming\Typora\typora-user-images\image-20241119211005310.png)

![image-20241119211416161](C:\Users\Chihao Shen\AppData\Roaming\Typora\typora-user-images\image-20241119211416161.png)

![image-20241119231138176](C:\Users\Chihao Shen\AppData\Roaming\Typora\typora-user-images\image-20241119231138176.png)