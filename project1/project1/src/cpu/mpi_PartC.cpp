//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void bfilter(unsigned char* output_pixels, JpegSOA input_image, int width,
             int num_channels, int start_line, int end_line, int offset);

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // MPI configuration
    MPI_Init(&argc, &argv);
    int numtasks, taskid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;

    int line_num = height - 2;
    int line_per_task = line_num / numtasks;
    int line_left = line_num % numtasks;

    int cuts[numtasks + 1];
    cuts[0] = 1;

    for (int i = 1; i <= numtasks; i++)
    {
        cuts[i] = cuts[i - 1] + line_per_task + (i <= line_left ? 1 : 0);
    }

    if (taskid == MASTER)
    {
        unsigned char* output_pixels =
            new unsigned char[width * height * num_channels];

        auto start_time = std::chrono::high_resolution_clock::now();
        /**
         * TODO: MPI PartC
         */

        bfilter(output_pixels, input_jpeg, width, num_channels, cuts[taskid],
                cuts[taskid + 1], 0);

        std::vector<MPI_Request> requests(numtasks - 1);
        for (int i = MASTER + 1; i < numtasks; i++)
        {
            unsigned char* start_pos =
                output_pixels + cuts[i] * width * num_channels;

            int length = (cuts[i + 1] - cuts[i]) * width * num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD,
                     &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        // Save output
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JpegAOS output_jpeg{(Pixel*)output_pixels, width, height, num_channels,
                            input_jpeg.color_space};
        if (export_jpeg(output_jpeg, output_filepath))
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

        delete[] input_jpeg.r_values;
        delete[] input_jpeg.g_values;
        delete[] input_jpeg.b_values;
        delete[] output_pixels;

        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds\n";
    }
    else
    {
        int length = width * (cuts[taskid + 1] - cuts[taskid]) * num_channels;
        int offset = width * cuts[taskid];

        unsigned char* output_pixels = new unsigned char[length];
        memset(output_pixels, 0, length);

        bfilter(output_pixels, input_jpeg, width, num_channels, cuts[taskid],
                cuts[taskid + 1], offset);

        MPI_Send(output_pixels, length, MPI_CHAR, MASTER, TAG_GATHER,
                 MPI_COMM_WORLD);

        delete[] output_pixels;
    }

    MPI_Finalize();
    return 0;
}

void bfilter(unsigned char* output_pixels, JpegSOA input_image, int width,
             int num_channels, int start_line, int end_line, int offset)
{
    for (int channel = 0; channel < num_channels; ++channel)
    {
        for (int y = start_line; y < end_line; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int id = y * width + x;
                ColorValue filtered_value = bilateral_filter(
                    input_image.get_channel(channel), y, x, width);
                output_pixels[(id - offset) * num_channels + channel] =
                    filtered_value;
            }
        }
    }
}
