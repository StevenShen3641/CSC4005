#include "ops.hpp"
#include <openacc.h>
const float epsilon = 1e-20;

void gemm(const float *A, const float *B, float *Out, size_t batch, size_t mn, size_t k)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop collapse(2) copyin(A[0 : batch * mn], B[0 : mn * k]) copyout(Out[0 : batch * k])
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t i = 0; i < k; ++i)
        {
            Out[b * k + i] = 0.0f;
            for (size_t j = 0; j < mn; ++j)
            {
                Out[b * k + i] += A[b * mn + j] * B[j * k + i];
            }
        }
    }
    // END YOUR CODE HERE <-
}

void add_bias(float *A, float *B, const float *bias, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : batch * out_dim], bias[0 : out_dim]) copyout(B[0 : batch * out_dim])
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t i = 0; i < out_dim; ++i)
        {
            B[b * out_dim + i] = A[b * out_dim + i] + bias[i];
        }
    }
    // END YOUR CODE HERE <-
}

void Relu(float *A, float *B, size_t size)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : size]) copyout(B[0 : size])
    for (size_t i = 0; i < size; ++i)
    {
        B[i] = (A[i] > 0) ? A[i] : 0;
    }
    // END YOUR CODE HERE <-
}

void Softmax(float *A, float *B, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : batch * out_dim]) copyout(B[0 : batch * out_dim])
    for (size_t b = 0; b < batch; ++b)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < out_dim; ++i)
        {
            sum += exp(A[b * out_dim + i]);
        }
        for (size_t i = 0; i < out_dim; ++i)
        {
            B[b * out_dim + i] = exp(A[b * out_dim + i]) / sum;
        }
    }
    // END YOUR CODE HERE <-
}

void vector_to_one_hot_matrix(const unsigned char *A, float *B, size_t batch, size_t out_dim)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : batch]) copyout(B[0 : batch * out_dim])
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t i = 0; i < out_dim; ++i)
            B[b * out_dim + i] = (A[b] == i) ? 1.0f : 0.0f;
    }
    // END YOUR CODE HERE <-
}

void cross_entropy_loss(const float *A, const float *B, float *Loss, size_t batch, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
    // Optional for your debug
    // END YOUR CODE HERE <-
}

void cross_entropy_loss_grad(const float *A, const float *B, float *Grad, size_t batch, size_t out_dim)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : batch * out_dim], B[0 : batch * out_dim]) copyout(Grad[0 : batch * out_dim])
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t i = 0; i < out_dim; ++i)
            Grad[b * out_dim + i] = A[b * out_dim + i] - B[b * out_dim + i]; // predict - real
    }
    // END YOUR CODE HERE <-
}

void update_bias(float *Bias, const float *Output_Grad, size_t batch, float lr, size_t out_dim)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(Output_Grad[0 : batch * out_dim]) copyout(Bias[0 : out_dim])
    for (size_t i = 0; i < out_dim; ++i)
    {
        float grad_sum = 0.0f;
#pragma acc loop reduction(+ : grad_sum)
        for (size_t b = 0; b < batch; ++b)
            grad_sum += Output_Grad[b * out_dim + i];
        Bias[i] -= lr * grad_sum;
    }

    // END YOUR CODE HERE <-
}

void input_grad(const float *Weight, const float *Output_Grad, float *Input, float *Input_Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
#pragma acc parallel loop collapse(2) copyin(Weight[0 : in_dim * out_dim], Output_Grad[0 : batch * out_dim], Input[0 : batch * in_dim]) copyout(Input_Grad[0 : batch * in_dim])
    for (size_t b = 0; b < batch; ++b)
    {
        for (size_t i = 0; i < in_dim; ++i)
        {
            Input_Grad[b * in_dim + i] = 0.0f;
            for (size_t j = 0; j < out_dim; ++j)
            {
                Input_Grad[b * in_dim + i] += Output_Grad[b * out_dim + j] * Weight[i * out_dim + j];
            }
        }
    }
    // END YOUR CODE HERE <-
}

void update_weight(float *Weight, const float *Output_Grad, const float *Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    // BEGIN YOUR CODE HERE ->
#pragma acc parallel loop collapse(2) copyin(Output_Grad[0 : batch * out_dim], Input[0 : batch * in_dim]) copyout(Weight[0 : in_dim * out_dim])
    for (size_t i = 0; i < in_dim; ++i)
    {

        for (size_t j = 0; j < out_dim; ++j)
        {
            float grad_sum = 0.0f;
#pragma acc loop reduction(+ : grad_sum)
            for (size_t b = 0; b < batch; ++b)
            {
                grad_sum += Output_Grad[b * out_dim + j] * Input[b * in_dim + i];
            }
            Weight[i * out_dim + j] -= lr * grad_sum;
        }
    }
    // END YOUR CODE HERE <-
}

void relu_grad(const float *A, float *Grad, size_t batch, size_t out_dim)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : batch * out_dim]) copyout(Grad[0 : batch * out_dim])
    for (auto i = 0; i < batch * out_dim; ++i)
    {
        Grad[i] = (A[i] > 0) ? Grad[i] : 0;
    }

    // END YOUR CODE HERE <-
}

float mean_acc(const unsigned char *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE HERE ->
    int correct = 0;
#pragma acc parallel loop copyin(result[0 : images_num], labels_array[0 : images_num])
    for (size_t i = 0; i < images_num; ++i)
    {
        if (result[i] == labels_array[i])
        {
            ++correct;
        }
    }
    return correct / float(images_num);
    // END YOUR CODE HERE <-
}

void argmax(const float *A, unsigned char *B, size_t num_classes, size_t images_num)
{
// BEGIN YOUR CODE HERE ->
#pragma acc parallel loop copyin(A[0 : images_num * num_classes]) copyout(B[0 : images_num])
    for (size_t i = 0; i < images_num; ++i)
    {
        unsigned char max_idx = 0;
        float max_val = A[i * num_classes];
        for (size_t n = 1; n < num_classes; ++n)
        {
            if (A[i * num_classes + n] > max_val)
            {
                max_val = A[i * num_classes + n];
                max_idx = n;
            }
        }
        B[i] = max_idx;
    }
    // END YOUR CODE HERE <-
}
