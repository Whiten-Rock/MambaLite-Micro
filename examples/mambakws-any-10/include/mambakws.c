/**
 * @file mambakws.c
 * @brief mambakws model
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#define B 1
#define T 100
#define F 40
#define H 64
#define C 10
int mambakws_forward(float *input)
{
    Tensor *mamba_input = tensor_zeros(1, 100, 64);
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                float sum =linear_in_bias[h];
                for (int f = 0; f < F; f++)
                {
                    int input_idx = (b * T + t) * F + f;
                    int weight_idx = h * F + f;
                    sum += input[input_idx] * linear_in_weight_matrix.data[weight_idx];
                }
                int output_idx = (b * T + t) * H + h;
                mamba_input->data[output_idx] = sum;
            }
        }
    }
    Mamba_Variables *model = Mamba_Init();
    Tensor *output = tensor_zeros(1, 100, 64);
    Mamba_Forward(model, mamba_input, output);
    Tensor *output_T = tensor_zeros(1, 100, 64);
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                int in_idx = (b * T + t) * H + h;
                int out_idx = (b * H + h) * T + t;
                output_T->data[out_idx] = output->data[in_idx];
            }
        }
    }

    Matrix *pooled = matrix_zeros(B, H);
    for (int b = 0; b < B; b++)
    {
        for (int h = 0; h < H; h++)
        {
            float sum = 0.0f;
            for (int t = 0; t < T; t++)
            {
                int idx = (b * H + h) * T + t; // [B, H, T]
                sum += output_T->data[idx];
            }
            pooled->data[b * H + h] = sum / T;
        }
    }
    Matrix *y = matrix_zeros(B, C);
    for (int b_idx = 0; b_idx < B; b_idx++)
    {
        for (int c = 0; c < C; c++)
        {
            float sum = classifier_bias[c];
            for (int h = 0; h < H; h++)
            {
                sum += pooled->data[b_idx * H + h] * classifier_weight_matrix.data[c * H + h];
            }
            y->data[b_idx * C + c] = sum;
            printf("%f", sum);
        }
    }
    int max_idx = 0;
    for (int b = 0; b < B; b++)
    {
        max_idx = 0;
        float max_val = y->data[b * C];
        for (int c = 1; c < C; c++)
        {
            float val = y->data[b * C + c];
            if (val > max_val)
            {
                max_val = val;
                max_idx = c;
            }
        }
    }
    return max_idx;
}