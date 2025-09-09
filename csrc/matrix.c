/**
 * @file matrix.c
 *
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once
#include <string.h>
#include <math.h>
#include "matrix.h"
#define MAT_ELEMENT(mat, row, col) ((mat)->data[(row)*(mat)->cols + (col)])

MatError matrix_init(Matrix* mat, uint16_t rows, uint16_t cols, MatType* buffer)
{
    if (!buffer) return MAT_ERR_MEMORY;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = buffer;
    return MAT_OK;
}

// Generate Identity matrix
Matrix* matrix_identity(uint16_t rows)
{
    Matrix* mat = matrix_zeros(rows,rows);
    for (uint16_t i = 0; i < rows; i++) 
    {
       mat->data[i*rows+i]=1;
    }
    return mat;
}
// Generate zero matrix, allocate new sapce
Matrix* matrix_zeros(uint16_t rows, uint16_t cols)
{
    size_t data_size = rows * cols * sizeof(MatType);
    Matrix* mat =(Matrix *) malloc(sizeof(Matrix) + data_size);
    if(!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (MatType*)(mat + 1);
    size_t total = rows * cols;
    for (uint16_t i = 0; i < total; i++) {
        mat->data[i] =0;
    }
    return mat;
}

void matrix_free(Matrix* mat) 
{
    free(mat);
}

MatError matrix_add(const Matrix* a, const Matrix* b, Matrix* result)
{
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols)
        return MAT_ERR_DIM_MISMATCH;

    const uint16_t total = a->rows * a->cols;
    for (uint16_t i = 0; i < total; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return MAT_OK;
}
MatError matrix_sub(const Matrix* a, const Matrix* b, Matrix* result)
{
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols)
        return MAT_ERR_DIM_MISMATCH;

    const uint16_t total = a->rows * a->cols;
    for (uint16_t i = 0; i < total; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return MAT_OK;
}

MatError matrix_mul(const Matrix* a, const Matrix* b, Matrix* result) {
    if (a->cols != b->rows || result->rows != a->rows || result->cols != b->cols)
        return MAT_ERR_DIM_MISMATCH;

    for (uint16_t i = 0; i < a->rows; i++) {
        for (uint16_t j = 0; j < b->cols; j++) {
            MatType sum = 0;
            for (uint16_t k = 0; k < a->cols; k++) {
                sum += MAT_ELEMENT(a, i, k) * MAT_ELEMENT(b, k, j);
            }
            MAT_ELEMENT(result, i, j) = sum;
        }
    }
    return MAT_OK;
}

MatError matrix_copy(Matrix* a,Matrix* result)
{
    const uint16_t total = a->rows * a->cols;
    for (uint16_t i = 0; i < total; i++) {
        result->data[i] = a->data[i];
    }
    return MAT_OK;
}

MatError matrix_sigmoid(Matrix* mat, Matrix* result)
{
    if (mat->rows != result->rows || mat->cols != result->cols) {
        return MAT_ERR_DIM_MISMATCH;
    }
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        result->data[i] = (MatType)(1.0 / (1.0 + exp(-mat->data[i])));
    }
    return MAT_OK;
}

MatError matrix_hadamard_prod(Matrix* a, Matrix* b, Matrix* result) 
{
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != result->rows || a->cols != result->cols) {
        return MAT_ERR_DIM_MISMATCH;
    }

    for (int i = 0; i < a->rows * a->cols; i++) {
        result->data[i] = result->data[i] * result->data[i];
    }
    return MAT_OK;
}