/**
 * @file main.cpp
 * @brief Sample mambahar firmware for STM32H7
 * Copyright (c) 2025 MambaLite-Micro Authors
 * Licensed under the MIT License.
 */
#include <Arduino.h>
#include <stdio.h>
#include <inttypes.h>
#include "task.h"
extern "C"
{
#include "../../../csrc/matrix.c"
#include "../../../csrc/tensor.c"
#include "../../../csrc/mamba.c"
#include "mambahar.c"
#include "sample_input.h"
}

static inline uint64_t micros64() {
  return (uint64_t)micros();
}
static inline uint64_t us_diff(uint64_t start, uint64_t end) {
  if (end >= start) return end - start;
  return ((1ULL << 32) - start) + end;
}

void setup() {
  Serial.begin(115200);

  for (int i = 5; i >= 0; --i) {
    delay(1000);
  }

  uint64_t t0 = micros64();
  int pred = mambahar_forward(input_data);
  uint64_t t1 = micros64();
  uint64_t elapsed_us = us_diff(t0, t1);
  Serial.println("Model_Forward completed.");
  Serial.print("Prediction: ");
  Serial.println(pred);
  Serial.print("Elapsed time:");
  Serial.println(elapsed_us);

  for (int i = 10; i >= 0; --i) {
    delay(1000);
  }
  Serial.println("Restarting now.");
  delay(10);
  NVIC_SystemReset();
}

void loop(){};