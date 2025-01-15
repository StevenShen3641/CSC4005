import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import cv2
import time

elapsed_time_ms = 0


@triton.jit
def bilateral_filter_kernel(input_ptr, output_ptr, ksize, height, width, sigma_space, sigma_density):
    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    channel = tl.program_id(axis=2)

    if ((pid_w == 0) or (pid_w == width)) or ((pid_h == 0) or (pid_h == height)):
        return

    output_index = (pid_h * width + pid_w) * 3 + channel
    result = 0.
    total_weight = 0.
    half_ksize = ksize // 2
    for dy in range(-half_ksize, half_ksize + 1):
        for dx in range(-half_ksize, half_ksize + 1):
            neighbor_h = pid_h + dy
            neighbor_w = pid_w + dx
            if ((1 <= neighbor_h) and (neighbor_h < height)) and ((0 <= neighbor_w) and (neighbor_w < width)):
                input_index = (neighbor_h * width + neighbor_w) * 3 + channel
                neighbor_value = tl.load(input_ptr + input_index)

                center_index = (pid_h * width + pid_w) * 3 + channel
                temp = neighbor_value - tl.load(input_ptr + center_index)
                spatial_weight = tl.exp(-(dy * dy + dx * dx) / (2 * sigma_space * sigma_space))
                density_weight = tl.exp(-(temp * temp) / (2 * sigma_density * sigma_density))
                weight = spatial_weight * density_weight
                result += neighbor_value * weight
                total_weight += weight

    if total_weight > 0:
        result /= total_weight

    tl.store(output_ptr + output_index, result)


def bfilter(img_tensor, ksize):
    sigma_space = 1.7
    sigma_density = 50.0
    assert img_tensor.is_contiguous(), "Matrix A must be contiguous"
    height, width, _ = img_tensor.shape

    img_tensor = img_tensor.clone().detach().to(device="cuda", dtype=torch.float32)

    output_tensor = torch.empty((height, width, 3), device=img_tensor.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(height, 1), triton.cdiv(width, 1), 3)

    start = time.time()
    bilateral_filter_kernel[grid](img_tensor, output_tensor, ksize, height, width, sigma_space, sigma_density)
    end = time.time()
    global elapsed_time_ms
    elapsed_time_ms = end * 1000 - start * 1000

    return output_tensor


def main(input_image_path, output_image_path):
    ksize = 3
    print(f"Input file from: {input_image_path}")
    img = cv2.imread(input_image_path).astype(np.float32)
    img_tensor = torch.tensor(img, device="cuda", dtype=torch.float32)

    output_tensor = bfilter(img_tensor, ksize)

    cv2.imwrite(output_image_path, output_tensor.cpu().numpy())
    print(f"Output file to: {output_image_path}")

    print(f"Transformation Complete!")
    print(f"Execution Time: {elapsed_time_ms:.2f} milliseconds")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
