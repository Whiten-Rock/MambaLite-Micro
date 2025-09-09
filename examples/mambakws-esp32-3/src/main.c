/**
 * @file main.c
 * @brief Sample mambakws firmware for ESP32S3
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#include <stdio.h>
#include <inttypes.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "../../../csrc/matrix.c"
#include "../../../csrc/mamba.c"
#include "mambakws.c"
#include "sample_input.h"

void app_main(void)
{
    for (int i = 5; i >= 0; i--) {
        printf("starting in %d seconds...\n", i);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    int64_t start_time = esp_timer_get_time();
    int pred= mambakws_forward(input_data);
    int64_t end_time = esp_timer_get_time();
    printf("Model_Forward completed.\n");
    switch (pred)
    {
    case 0:
        printf("predicted label:_unknown_\n");
        break;
    case 1:
        printf("predicted label:no\n");
        break;
    case 2:
        printf("predicted label:yes\n");
        break;
    default:
        break;
    }
    printf("Elapsed time: %lld microseconds (%.2f milliseconds)\n", 
           end_time - start_time, 
           (end_time - start_time) / 1000.0);

    for (int i = 10; i >= 0; i--) {
        printf("Restarting in %d seconds...\n", i);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    printf("Restarting now.\n");
    fflush(stdout);
    esp_restart();
}
