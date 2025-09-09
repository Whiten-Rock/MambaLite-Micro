/**
 * @file matrix.h
 *
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once
#ifndef EMBEDDED_MATRIX_H
#define EMBEDDED_MATRIX_H
#include <stdint.h>
#ifndef MatType
typedef float MatType;
#endif
typedef enum {
    MAT_OK,
    MAT_ERR_DIM_MISMATCH,
    MAT_ERR_MEMORY,
    MAT_ERR_SINGULAR
} MatError;

typedef struct {
    uint16_t rows;
    uint16_t cols;
    MatType* data;
} Matrix;

MatError matrix_init(Matrix* mat, uint16_t rows, uint16_t cols, MatType* buffer);
MatError matrix_add(const Matrix* a, const Matrix* b, Matrix* result);
MatError matrix_sub(const Matrix* a, const Matrix* b, Matrix* result);
MatError matrix_mul(const Matrix* a, const Matrix* b, Matrix* result);

MatError matrix_sigmoid(Matrix* mat, Matrix* result);
MatError matrix_hadamard_prod(Matrix* a, Matrix* b, Matrix* result);

Matrix* matrix_zeros(uint16_t rows, uint16_t cols);
Matrix* matrix_identity(uint16_t rows);


void matrix_free(Matrix* mat);
#endif