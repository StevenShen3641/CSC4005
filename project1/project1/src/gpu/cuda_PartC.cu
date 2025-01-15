//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"

/**
 * Demo kernel device function to clamp pixel value
 *
 * You may mimic this to implement your own kernel device functions
 */
__device__ ColorValue d_bilateral_filter(const ColorValue* values, int row,
                                         int col, int width);
__device__ ColorValue d_clamp_pixel_value(float value);
__global__ void apply_filter_kernel(Pixel* input_pixels, Pixel* output_pixels,
                                    int width, int height, int num_channels);
__device__ ColorValue d_get_channel(Pixel pixel, int channel);

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image in aos form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegAOS input_jpeg = read_jpeg_aos(input_filename);
    if (input_jpeg.pixels == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;

    unsigned char* output_pixels = new unsigned char[buffer_size];

    /**
     * TODO: CUDA PartC
     */
    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc((void**)&d_input, buffer_size);
    cudaMalloc((void**)&d_output, buffer_size);
    cudaMemset(d_output, 0, buffer_size);

    cudaMemcpy(d_input, input_jpeg.pixels, buffer_size, cudaMemcpyHostToDevice);

    // Set CUDA grid and block sizes
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    apply_filter_kernel<<<gridDim, blockDim>>>(
        (Pixel*)d_input, (Pixel*)d_output, width, height, num_channels);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuDuration, start, stop);
    cudaMemcpy(output_pixels, d_output, buffer_size, cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegAOS output_jpeg{(Pixel*)output_pixels, width, height, num_channels,
                        input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.pixels;
    delete[] output_pixels;
    // Release GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

__device__ ColorValue d_bilateral_filter(const Pixel* pixels, int row, int col,
                                         int width, int channel)
{
    ColorValue value_11 =
        d_get_channel(pixels[(row - 1) * width + (col - 1)], channel);
    ColorValue value_12 =
        d_get_channel(pixels[(row - 1) * width + col], channel);
    ColorValue value_13 =
        d_get_channel(pixels[(row - 1) * width + (col + 1)], channel);
    ColorValue value_21 =
        d_get_channel(pixels[row * width + (col - 1)], channel);
    ColorValue value_22 = d_get_channel(pixels[row * width + col], channel);
    ColorValue value_23 =
        d_get_channel(pixels[row * width + (col + 1)], channel);
    ColorValue value_31 =
        d_get_channel(pixels[(row + 1) * width + (col - 1)], channel);
    ColorValue value_32 =
        d_get_channel(pixels[(row + 1) * width + col], channel);
    ColorValue value_33 =
        d_get_channel(pixels[(row + 1) * width + (col + 1)], channel);
    // Spatial Weights
    float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
    float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));
    // Intensity Weights
    ColorValue center_value = value_22;
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
    return d_clamp_pixel_value(filtered_value);
}

__device__ ColorValue d_clamp_pixel_value(float value)
{
    return value > 255 ? 255 : value < 0 ? 0 : static_cast<ColorValue>(value);
}

__global__ void apply_filter_kernel(Pixel* input_pixels, Pixel* output_pixels,
                                    int width, int height, int num_channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int id = (y * width + x);

        ColorValue filtered_value_r =
            d_bilateral_filter(input_pixels, y, x, width, 0);
        output_pixels[id].r = filtered_value_r;
        ColorValue filtered_value_g =
            d_bilateral_filter(input_pixels, y, x, width, 1);
        output_pixels[id].g = filtered_value_g;
        ColorValue filtered_value_b =
            d_bilateral_filter(input_pixels, y, x, width, 2);
        output_pixels[id].b = filtered_value_b;
    }
}

__device__ ColorValue d_get_channel(Pixel pixel, int channel)
{
    switch (channel)
    {
        case 0: return pixel.r;
        case 1: return pixel.g;
        case 2: return pixel.b;
        default: return 0;
    }
}