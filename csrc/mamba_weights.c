/**
 * @file mamba_weights.c
 * @brief mamba weight file
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#ifndef MAMBA_WEIGHTS_H
#include "mamba_weights.h"
#endif
#include "matrix.h"

Matrix linear_in_weight_matrix = {
    .rows = linear_in_weight_rows,
    .cols = linear_in_weight_cols,
    .data = (MatType*)linear_in_weight
};

Matrix mamba_A_log_matrix = {
    .rows = mamba_A_log_rows,
    .cols = mamba_A_log_cols,
    .data = (MatType*)mamba_A_log
};

Matrix mamba_in_proj_weight_matrix = {
    .rows = mamba_in_proj_weight_rows,
    .cols = mamba_in_proj_weight_cols,
    .data = (MatType*)mamba_in_proj_weight
};

Matrix mamba_conv1d_weight_matrix = {
    .rows = mamba_conv1d_weight_rows,
    .cols = mamba_conv1d_weight_cols,
    .data = (MatType*)mamba_conv1d_weight
};

Matrix mamba_x_proj_weight_matrix = {
    .rows = mamba_x_proj_weight_rows,
    .cols = mamba_x_proj_weight_cols,
    .data = (MatType*)mamba_x_proj_weight
};

Matrix mamba_dt_proj_weight_matrix = {
    .rows = mamba_dt_proj_weight_rows,
    .cols = mamba_dt_proj_weight_cols,
    .data = (MatType*)mamba_dt_proj_weight
};

Matrix mamba_out_proj_weight_matrix = {
    .rows = mamba_out_proj_weight_rows,
    .cols = mamba_out_proj_weight_cols,
    .data = (MatType*)mamba_out_proj_weight
};

Matrix classifier_weight_matrix = {
    .rows = classifier_weight_rows,
    .cols = classifier_weight_cols,
    .data = (MatType*)classifier_weight
};