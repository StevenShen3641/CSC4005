//
// Created by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #3: Parallel Sorting with Regular Sampling using MPI
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <mpi.h>

#include "../utils.hpp"

#define MASTER 0

/**
 * The following are variables for each process
 * Feel free to delete them or create new ones 
 */
std::vector<int> DATA;				  // Input dataset for sorting
std::vector<int> SortedData;		  // Output sorted data
std::vector<int> localData;			  // Data that each process gets
std::vector<int> localRegularSamples; // Regular samples of each process
std::vector<int> regularSamples;	  // Global regular samples in master
std::vector<int> pivots;			  // Pivots for load balancing
std::vector<int> splitters;			  // splitting indices
std::vector<int> mergedArray;		  // locally merged array
std::vector<int> lengths;			  // lengths of splitted array pieces
std::vector<int> obtainedKeys;		  // obtained keys from other processors

DistType DIST_TYPE;		  // Distribution type of the input dataset
int obtainedKeysSize = 0; // data size obtained from other processes
int dataCurrentProc;	  // Data Size the Current Process Gets
int T;					  // Total number of processes for sorting
int SIZE;				  // Size of the dataset to sort
int rank;				  // Rank of the current process

// data distribution phase
void phase_0()
{
	// regular data size that a processor will get
	int dataPerProc = std::ceil(SIZE / T);
	// the actual data size that the processor will get
	dataCurrentProc = (rank == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
	// allocate sufficient memory for the local array
	localData = std::vector<int>(dataCurrentProc, 0);
	std::vector<int> lenEachProc(T, 0); // Array size for each process
	std::vector<int> displacements(T);	// Displacement index for each proc
	if (rank == MASTER)
	{
		DATA = genRandomVec(SIZE, DIST_TYPE); // use default seed
		for (int index = 0; index < T; index++)
		{
			lenEachProc[index] = (index == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
		}
		displacements = prefixSum(lenEachProc);
	}
	// Scatter dataset to each processor
	MPI_Scatterv(&DATA[0], &lenEachProc[0], &displacements[0], MPI_INT, &localData[0], dataCurrentProc, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * TODO: local sorting and regular sampling phase
 * 1. You need to sort the local partition by any algorithm you want
 * 2. You need to pick T local regular samples
 */
void phase_1()
{
	/* Your codes here! */
	std::sort(localData.begin(), localData.end());
	int step = dataCurrentProc / T;
	for (int i = 0; i < T; i++)
		localRegularSamples.push_back(localData[i * step]);
}

/**
 * TODO: Pivot Selection Phase
 * 1. Gather all the local samples from each processstd::sort(localData.begin(), localData.end());// Pick T evenly spaced samplesint step = dataCurrentProc / T;for (int i = 0; i < T; i++)localRegularSamples.push_back(localData[i * step]);
 * 2. Select (T - 1) global pivots
 */
void phase_2()
{
	/* Your codes here! */
	regularSamples.resize(T * T);
	MPI_Gather(localRegularSamples.data(), T, MPI_INT, regularSamples.data(), T, MPI_INT, MASTER, MPI_COMM_WORLD);

	if (rank == MASTER)
	{
		std::sort(regularSamples.begin(), regularSamples.end());
		for (int i = 1; i < T; i++)
			pivots.push_back(regularSamples[i * T]);
	}
	pivots.resize(T - 1);
	MPI_Bcast(pivots.data(), T - 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * TODO: Split the data pieces and exchange them across processes
 */
void phase_3()
{
	/* Your codes here! */
    splitters.clear();
    size_t pivotIndex = 0;
    for (int i = 0; i < dataCurrentProc; i++)
    {
        if (pivotIndex < pivots.size() && localData[i] > pivots[pivotIndex])
        {
            splitters.push_back(i);
            pivotIndex++;
        }
    }
    while (splitters.size() < T - 1)
        splitters.push_back(dataCurrentProc);

    splitters.push_back(dataCurrentProc);

    std::vector<int> sendCounts(T);
    std::vector<int> sendOffsets(T, 0);
    for (int i = 0; i < T; i++)
        sendCounts[i] = (i == 0 ? splitters[i] : splitters[i] - splitters[i - 1]);

    for (int i = 1; i < T; i++)
        sendOffsets[i] = sendOffsets[i - 1] + sendCounts[i - 1];

    std::vector<int> recvCounts(T, 0);
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int totalRecvSize = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
    obtainedKeys.resize(totalRecvSize);

    std::vector<int> recvOffsets(T, 0);
    for (int i = 1; i < T; i++)
        recvOffsets[i] = recvOffsets[i - 1] + recvCounts[i - 1];

    MPI_Alltoallv(localData.data(), sendCounts.data(), sendOffsets.data(), MPI_INT,
                  obtainedKeys.data(), recvCounts.data(), recvOffsets.data(), MPI_INT, MPI_COMM_WORLD);
}

/**
 * TODO: Merge local partitions
 * You can use k-way merge in Task #2 if you want
 */
void phase_4()
{
	/* Your codes here! */
	std::sort(obtainedKeys.begin(), obtainedKeys.end());
	mergedArray = obtainedKeys;
}

/**
 * TODO: Merge all local arrays into SortedData in master process
 */
void phase_merge()
{
	/* Your codes here! */
    std::vector<int> recvCounts(T, 0);
    std::vector<int> displacements(T, 0);
    int localMergedSize = mergedArray.size();

    MPI_Gather(&localMergedSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
    {
        int totalSize = std::accumulate(recvCounts.begin(), recvCounts.end(), 0);
        SortedData.resize(totalSize);

        for (int i = 1; i < T; i++)
            displacements[i] = displacements[i - 1] + recvCounts[i - 1];
    }

    MPI_Gatherv(mergedArray.data(), localMergedSize, MPI_INT,
                SortedData.data(), recvCounts.data(), displacements.data(), MPI_INT, MASTER, MPI_COMM_WORLD);

}

/**
 * You can measure the time of each phase with this function
 */
void measureTime(void (*fun)(), char *processorName, char *title, int shouldLog)
{
	if (shouldLog)
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		fun();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time - start_time);
		printf("[%s:%d] %s took %ld ms\n", processorName, rank, title, elapsed_time.count());
	}
	else
	{
		fun();
	}
}

int main(int argc, char *argv[])
{
	// Verify input argument format
	if (argc != 3)
	{
		throw std::invalid_argument(
			"Invalid argument, should be: ./executable dist_type vector_size\n");
	}
	DIST_TYPE = str_2_dist_type(std::string(argv[1]));
	SIZE = atoi(argv[2]); // data size to sort

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &T);	  // how many processors are available
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // what's my rank?
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len); // What's processor name

	// Phase 0: Data distribution
	measureTime(phase_0, processor_name, "Phase 0", rank == 0);
	MPI_Barrier(MPI_COMM_WORLD);

	auto start_time = std::chrono::high_resolution_clock::now();

	// PHASE 1
    // For now, measureTime function does not print the time consumption for each phase, change the last param to 1 if you want to print it out
	measureTime(phase_1, processor_name, "Phase 1", 0);
	// PHASE 2
	measureTime(phase_2, processor_name, "Phase 2", 0);
	// PHASE 3
	measureTime(phase_3, processor_name, "Phase 3", 0);
	// PHASE 4
	measureTime(phase_4, processor_name, "Phase 4", 0);

	MPI_Barrier(MPI_COMM_WORLD);

	// PHASE Merge
	measureTime(phase_merge, processor_name, "Phase Merge", rank == 0);

	if (rank == MASTER)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time - start_time);
		std::cout << "Sorting Complete!" << std::endl;
		std::cout << "Execution Time: " << elapsed_time.count() << " 		milliseconds" << std::endl;
		checkSortResult(DATA, SortedData); // check if sorted
	}

	MPI_Finalize();
	return 0;
}
