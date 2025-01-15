//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

void* bfilter(void* arg);
struct ThreadData
{
    JpegSOA input_image;
    JpegSOA output_image;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int NUM_THREADS = std::stoi(argv[3]);

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];

    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    int line_per_task = (height - 2) / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i] = {
            input_jpeg,
            output_jpeg,
            width,
            height,
            num_channels,
            i * line_per_task + 1,
            (i == NUM_THREADS - 1) ? height : (i + 1) * line_per_task + 1};
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: Pthread PartC
     */

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, bfilter, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Save output
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    delete[] threads;
    delete[] threadData;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}

void* bfilter(void* arg)
{
    ThreadData* threadData = (ThreadData*)arg;
    for (int channel = 0; channel < threadData->num_channels; ++channel)
    {
        for (int y = threadData->start_row; y < threadData->end_row; y++)
        {
            for (int x = 1; x < threadData->width; x++)
            {
                int id = y * threadData->width + x;
                ColorValue filtered_value = bilateral_filter(
                    threadData->input_image.get_channel(channel), y, x, threadData->width);
                threadData->output_image.set_value(channel, id, filtered_value);
            }
        }
    }
    return nullptr;
}
