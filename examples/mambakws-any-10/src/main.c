/**
 * @file main.c
 * @brief Sample mambakws program without specific target backend
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include "../include/mamba_weights.h"
#include "../../../csrc/matrix.c"
#include "../../../csrc/tensor.c"
#include "../../../csrc/mamba.c"
#include "../include/mambakws.c"
#include "../include/sample_input.h"

int main()
{
    int pred= mambakws_forward(input_data);
    printf("\n%d",pred);
    getchar();
    return 0;
}