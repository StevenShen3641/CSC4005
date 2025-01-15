#!/bin/bash

mkdir -p ./perf_results

# Naive
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/naive_1024.data ./build/src/naive ./matrices/matrix5.txt ./matrices/matrix6.txt ./result.txt
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/naive_2048.data ./build/src/naive ./matrices/matrix7.txt ./matrices/matrix8.txt ./result.txt

# Locality
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/locality_1024.data ./build/src/locality ./matrices/matrix5.txt ./matrices/matrix6.txt ./result.txt
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/locality_2048.data ./build/src/locality ./matrices/matrix7.txt ./matrices/matrix8.txt ./result.txt

# SIMD
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/simd_1024.data ./build/src/simd ./matrices/matrix5.txt ./matrices/matrix6.txt ./result.txt
srun -n 1 --cpus-per-task 1 perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/simd_2048.data ./build/src/simd ./matrices/matrix7.txt ./matrices/matrix8.txt ./result.txt

# OpenMP
for threads in 1 2 4 8 16 32; do
    srun -n 1 --cpus-per-task ${threads} perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/openmp_1024_${threads}.data ./build/src/openmp $threads ./matrices/matrix5.txt ./matrices/matrix6.txt ./result.txt
    srun -n 1 --cpus-per-task ${threads} perf record -e cpu-cycles,cache-misses,page-faults -g -o ./perf_results/openmp_2048_${threads}.data ./build/src/openmp $threads ./matrices/matrix7.txt ./matrices/matrix8.txt ./result.txt
done

# MPI
for processes in 1 2 4 8 16 32; do
    threads=$((32 / processes))
    mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/mpi_1024_${processes}.data ./build/src/mpi $threads ./matrices/matrix5.txt ./matrices/matrix6.txt ./result.txt
    mpirun -np $processes perf stat -e cpu-cycles,cache-misses,page-faults -o ./perf_results/mpi_2048_${processes}.data ./build/src/mpi $threads ./matrices/matrix7.txt ./matrices/matrix8.txt ./result.txt
done
