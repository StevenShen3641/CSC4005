//
// Created by Liu Yuxuan on 2024/9/10
// Modified on Yang Yufan's simd_PartB.cpp on 2023/9/16
// Email: yufanyang1@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>

#include "../utils.hpp"

static inline float bfilter(ColorValue center, ColorValue value);
static float _reduce_sum(__m256 v);
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];


    
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};



    float w_spatial_border = expf(-0.5 / powf(SIGMA_D, 2));
    float w_spatial_corner = expf(-1.0 / powf(SIGMA_D, 2));

    // Ignore the centroid to speed up
    __m256 w_spatial = 
        _mm256_set_ps(w_spatial_corner, w_spatial_border, w_spatial_corner,
        w_spatial_border, w_spatial_border, w_spatial_corner, 
        w_spatial_border, w_spatial_corner);


    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: SIMD PartC
     */
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int id = y * width + x;
            ColorValue center_r = input_jpeg.r_values[id];
            ColorValue center_g = input_jpeg.g_values[id];
            ColorValue center_b = input_jpeg.b_values[id];
            ColorValue* start_pos_r = &input_jpeg.r_values[id - width - 1];
            ColorValue* start_pos_g = &input_jpeg.g_values[id - width - 1];
            ColorValue* start_pos_b = &input_jpeg.b_values[id - width - 1];

            __m256 w_intensity_r = _mm256_set_ps(
                bfilter(center_r, *start_pos_r),
                bfilter(center_r, *(start_pos_r + 1)),
                bfilter(center_r, *(start_pos_r + 2)),
                bfilter(center_r, *(start_pos_r + width)),
                bfilter(center_r, *(start_pos_r + width + 2)),
                bfilter(center_r, *(start_pos_r + 2 * width)),
                bfilter(center_r, *(start_pos_r + 2 * width + 1)),
                bfilter(center_r, *(start_pos_r + 2 * width + 2))
            );

            __m256 w_intensity_g = _mm256_set_ps(
                bfilter(center_g, *start_pos_g),
                bfilter(center_g, *(start_pos_g + 1)),
                bfilter(center_g, *(start_pos_g + 2)),
                bfilter(center_g, *(start_pos_g + width)),
                bfilter(center_g, *(start_pos_g + width + 2)),
                bfilter(center_g, *(start_pos_g + 2 * width)),
                bfilter(center_g, *(start_pos_g + 2 * width + 1)),
                bfilter(center_g, *(start_pos_g + 2 * width + 2))
            );

            __m256 w_intensity_b = _mm256_set_ps(
                bfilter(center_b, *start_pos_b),
                bfilter(center_b, *(start_pos_b + 1)),
                bfilter(center_b, *(start_pos_b + 2)),
                bfilter(center_b, *(start_pos_b + width)),
                bfilter(center_b, *(start_pos_b + width + 2)),
                bfilter(center_b, *(start_pos_b + 2 * width)),
                bfilter(center_b, *(start_pos_b + 2 * width + 1)),
                bfilter(center_b, *(start_pos_b + 2 * width + 2))
            );

            __m256 values_r = _mm256_set_ps((float)*start_pos_r, (float)*(start_pos_r + 1), (float)*(start_pos_r + 2),
            (float)*(start_pos_r + width), (float)*(start_pos_r + width + 2), (float)*(start_pos_r + 2 * width), 
            (float)*(start_pos_r + 2 * width + 1), (float)*(start_pos_r + 2 * width + 2));
            __m256 values_g = _mm256_set_ps((float)*start_pos_g, (float)*(start_pos_g + 1), (float)*(start_pos_g + 2),
            (float)*(start_pos_g + width), (float)*(start_pos_g + width + 2), (float)*(start_pos_g + 2 * width), 
            (float)*(start_pos_g + 2 * width + 1), (float)*(start_pos_g + 2 * width + 2));
            __m256 values_b = _mm256_set_ps((float)*start_pos_b, (float)*(start_pos_b + 1), (float)*(start_pos_b + 2),
            (float)*(start_pos_b + width), (float)*(start_pos_b + width + 2), (float)*(start_pos_b + 2 * width), 
            (float)*(start_pos_b + 2 * width + 1), (float)*(start_pos_b + 2 * width + 2));
            
            __m256 weights_r = _mm256_mul_ps(w_spatial, w_intensity_r);
            __m256 weights_g = _mm256_mul_ps(w_spatial, w_intensity_g);
            __m256 weights_b = _mm256_mul_ps(w_spatial, w_intensity_b);

            float sum_weights_r = _reduce_sum(weights_r) + 1;  // w_22
            float sum_weights_g = _reduce_sum(weights_g) + 1;  // w_22
            float sum_weights_b = _reduce_sum(weights_b) + 1;  // w_22
            float filtered_value_r = (_reduce_sum(_mm256_mul_ps(weights_r, values_r)) + center_r) / sum_weights_r;
            float filtered_value_g = (_reduce_sum(_mm256_mul_ps(weights_g, values_g)) + center_g) / sum_weights_g;
            float filtered_value_b = (_reduce_sum(_mm256_mul_ps(weights_b, values_b)) + center_b) / sum_weights_b;
            ColorValue pixel_r = clamp_pixel_value(filtered_value_r);
            ColorValue pixel_g = clamp_pixel_value(filtered_value_g);
            ColorValue pixel_b = clamp_pixel_value(filtered_value_b);

            output_jpeg.r_values[id] =  pixel_r;
            output_jpeg.g_values[id] =  pixel_g;
            output_jpeg.b_values[id] =  pixel_b;
        }
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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}

static float _reduce_sum(__m256 v) {
    // Using adj approach
    __m256 temp1 = _mm256_hadd_ps(v, v);
    __m256 temp2 = _mm256_hadd_ps(temp1, temp1);
    __m128 high = _mm256_extractf128_ps(temp2, 1);
    __m128 low = _mm256_castps256_ps128(temp2);

    __m128 sum = _mm_add_ps(low, high);
    return (float)_mm_cvtss_f32(sum);
}

static inline float bfilter(ColorValue center, ColorValue value) {
    return expf(powf(center - value, 2) / (-2 * powf(SIGMA_R, 2)));
}
