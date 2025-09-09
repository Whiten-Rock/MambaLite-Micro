/**
 * @file mamba.h
 * @brief mamba inference kernel head
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#pragma once
#include "matrix.h"
typedef enum {
    Mamba_OK,
    Mamba_ERR_DIM_MISMATCH,
    Mamba_ERR_MEMORY,
    Mamba_ERR_SINGULAR
} MambaError;
