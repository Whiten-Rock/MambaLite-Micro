/**
 * @file tensor.c
 *
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once
#include <string.h>
#include <math.h>
typedef float TensorType;
typedef struct {
    uint16_t d1;
    uint16_t d2;
    uint16_t d3;
    TensorType* data;
}Tensor;

Tensor* tensor_zeros(int d1,int d2,int d3)
{
    size_t data_size = d1 * d2 * d3 * sizeof(TensorType);
    Tensor* tensor = (Tensor *) malloc(sizeof(Tensor) + data_size);
    if(!tensor){
        printf("malloc failed");
        return NULL;
    }
    tensor->d1 = d1;
    tensor->d2 = d2;
    tensor->d3 = d3;
    tensor->data = (TensorType*)(tensor + 1);
    size_t total = d1 * d2 * d3;
    for (uint16_t i = 0; i < total; i++) {
        tensor->data[i] =0;
    }
    return tensor;
}

void tensor_free(Tensor* tensor)
{
    free(tensor);
}