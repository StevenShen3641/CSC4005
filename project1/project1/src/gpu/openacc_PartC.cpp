//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>
#include <stdio.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_get_channel(Pixel pixel, int channel)
{
    switch (channel)
    {
        case 0: return pixel.r;
        case 1: return pixel.g;
        case 2: return pixel.b;
        default: return 0;
    }
}

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255 : value < 0 ? 0 : static_cast<ColorValue>(value);
}

#pragma acc routine seq
ColorValue acc_bilateral_filter(const Pixel* pixels, int row, int col,
                                int width, int channel)
{
    int value_11 =
        acc_get_channel(pixels[(row - 1) * width + (col - 1)], channel);
    int value_12 = acc_get_channel(pixels[(row - 1) * width + col], channel);
    int value_13 =
        acc_get_channel(pixels[(row - 1) * width + (col + 1)], channel);
    int value_21 = acc_get_channel(pixels[row * width + (col - 1)], channel);
    int value_22 = acc_get_channel(pixels[row * width + col], channel);
    int value_23 = acc_get_channel(pixels[row * width + (col + 1)], channel);
    int value_31 =
        acc_get_channel(pixels[(row + 1) * width + (col - 1)], channel);
    int value_32 = acc_get_channel(pixels[(row + 1) * width + col], channel);
    int value_33 =
        acc_get_channel(pixels[(row + 1) * width + (col + 1)], channel);
    // Spatial Weights
    float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
    float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
    // Intensity Weights
    int center_value = value_22;
    float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_22 = 1.0;
    float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float w_33 = w_spatial_corner * expf(powf(center_value - value_33, 2) /
                                         (-2 * powf(SIGMA_R, 2)));
    float sum_weights =
        w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
    // Calculate filtered value
    float filtered_value =
        (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 + w_21 * value_21 +
         w_22 * center_value + w_23 * value_23 + w_31 * value_31 +
         w_32 * value_32 + w_33 * value_33) /
        sum_weights;
    return acc_clamp_pixel_value(filtered_value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegAOS input_jpeg = read_jpeg_aos(input_filename);
    if (input_jpeg.pixels == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    /**
     * TODO: OpenACC PartC
     */

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t channel_size = width * height;
    Pixel* input_pixels = new Pixel[channel_size];
    Pixel* output_pixels = new Pixel[channel_size];

    memset(output_pixels, 0, channel_size * num_channels);
    memcpy(input_pixels, input_jpeg.pixels, channel_size * num_channels);
    delete[] input_jpeg.pixels;

#pragma acc enter data copyin(input_pixels[0 : channel_size], \
                              output_pixels[0 : channel_size])

#pragma acc update device(input_pixels[0 : channel_size], \
                          output_pixels[0 : channel_size])
    auto start_time = std::chrono::high_resolution_clock::now();

#pragma acc parallel present(input_pixels[0 : channel_size], \
                             output_pixels[0 : channel_size]) num_gangs(1024)
    {
#pragma acc loop independent
        for (int y = 1; y < height - 1; y++)
        {
#pragma acc loop independent
            for (int x = 1; x < width - 1; x++)
            {
                int id = (y * width + x);
                ColorValue filtered_value_r =
                    acc_bilateral_filter(input_pixels, y, x, width, 0);
                output_pixels[id].r = filtered_value_r;
                ColorValue filtered_value_g =
                    acc_bilateral_filter(input_pixels, y, x, width, 1);
                output_pixels[id].g = filtered_value_g;
                ColorValue filtered_value_b =
                    acc_bilateral_filter(input_pixels, y, x, width, 2);
                output_pixels[id].b = filtered_value_b;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
#pragma acc update self(output_pixels[0 : channel_size])

#pragma acc exit data copyout(output_pixels[0 : channel_size])

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegAOS output_jpeg{output_pixels, width, height, num_channels,
                        input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_pixels;
    delete[] output_pixels;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
