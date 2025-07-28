#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "conv_weights.h"
#include "conv_biases.h"
#include "fc_weights.h"
#include "fc_biases.h"

// For performance metrics
#include <intrin.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>

#define IMG_SIZE 28
#define KERNEL_SIZE 3
#define NUM_KERNELS 8
#define STRIDE 1
#define OUT_SIZE (IMG_SIZE - KERNEL_SIZE + 1)
#define FC_IN (NUM_KERNELS * OUT_SIZE * OUT_SIZE)
#define FC_OUT 10

float conv_W[NUM_KERNELS][KERNEL_SIZE][KERNEL_SIZE];
float conv_B[NUM_KERNELS];

float fc_W[FC_OUT][FC_IN];
float fc_B[FC_OUT];

void load_conv_weights() {
    int idx = 0;
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                conv_W[k][i][j] = conv_weights[idx];
                idx++;
            }
        }
    }
    for (int i = 0; i < NUM_KERNELS; i++) {
        conv_B[i] = conv_biases[i];
    }
}

void load_fc_weights() {
    int idx = 0;
    for (int i = 0; i < FC_OUT; i++) {
        for (int j = 0; j < FC_IN; j++) {
            fc_W[i][j] = fc_weights[idx];
            idx++;
        }
    }
    for (int i = 0; i < FC_OUT; i++) {
        fc_B[i] = fc_biases[i];
    }
}

// ReLU activation
float relu(float x) {
    return x > 0 ? x : 0;
}

// 2D Convolution
void conv2d(float input[IMG_SIZE][IMG_SIZE], float output[NUM_KERNELS][OUT_SIZE][OUT_SIZE]) {
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int i = 0; i < OUT_SIZE; i++) {
            for (int j = 0; j < OUT_SIZE; j++) {
                float sum = 0.0;
                for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                    for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                        sum += input[i + ki][j + kj] * conv_W[k][ki][kj];
                    }
                }
                sum += conv_B[k];
                output[k][i][j] = relu(sum);
            }
        }
    }
}

// Flatten output
void flatten(float input[NUM_KERNELS][OUT_SIZE][OUT_SIZE], float output[NUM_KERNELS * OUT_SIZE * OUT_SIZE]) {
    int idx = 0;
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int i = 0; i < OUT_SIZE; i++) {
            for (int j = 0; j < OUT_SIZE; j++) {
                output[idx++] = input[k][i][j];
            }
        }
    }
}

// Fully connected layer
void fc(float input[NUM_KERNELS * OUT_SIZE * OUT_SIZE], float output[FC_OUT]) {
    for (int i = 0; i < FC_OUT; i++) {
        float sum = 0.0;
        for (int j = 0; j < NUM_KERNELS * OUT_SIZE * OUT_SIZE; j++) {
            sum += input[j] * fc_W[i][j];
        }
        sum += fc_B[i];
        output[i] = sum;
    }
}

// Predict function
int predict(float image[IMG_SIZE][IMG_SIZE],
    int is_performanceCheckMode,
    double *total_cycles_conv2d, 
    double *total_cycles_flatten, 
    double *total_cycles_fc
) {
    float conv_out[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
    float flat[NUM_KERNELS * OUT_SIZE * OUT_SIZE];
    float logits[FC_OUT];

    // Measure CPU cycles taken for each major operation
    LARGE_INTEGER conv2d_start, conv2d_end;
    LARGE_INTEGER flatten_start, flatten_end;
    LARGE_INTEGER fc_start, fc_end;
    
    // conv2d_start = clock();
    QueryPerformanceCounter(&conv2d_start);
    conv2d(image, conv_out);
    // conv2d_end = clock();
    QueryPerformanceCounter(&conv2d_end);

    // flatten_start = clock();
    QueryPerformanceCounter(&flatten_start);
    flatten(conv_out, flat);
    // flatten_end = clock();
    QueryPerformanceCounter(&flatten_end);

    // fc_start = clock();
    QueryPerformanceCounter(&fc_start);
    fc(flat, logits);
    // fc_end = clock();
    QueryPerformanceCounter(&fc_end);

    // Argmax
    int pred = 0;
    float max_val = logits[0];
    for (int i = 1; i < FC_OUT; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            pred = i;
        }
    }

    if (is_performanceCheckMode) {
        double total_ticks = (double)(conv2d_end.QuadPart - conv2d_start.QuadPart);
        total_ticks += (double)(flatten_end.QuadPart - flatten_start.QuadPart);
        total_ticks += (double)(fc_end.QuadPart - fc_start.QuadPart);
        // Add the PERCENTAGE of total ticks used to the accumulations.
        // NOTE: The reason we're tracking the proportions/ratios instead of tallying total ticks
        // Is because convolution operation uses SO many cycles that the counter variable will quickly
        // overflow :(
        *total_cycles_conv2d += (double)(conv2d_end.QuadPart - conv2d_start.QuadPart) / total_ticks;
        *total_cycles_flatten += (double)(flatten_end.QuadPart - flatten_start.QuadPart) / total_ticks;
        *total_cycles_fc += (double)(fc_end.QuadPart - fc_start.QuadPart) / total_ticks;
        // printf("Conv2d (cycles):\t%.2f\n", (double)(conv2d_end.QuadPart - conv2d_start.QuadPart) / total_ticks);
        // printf("Flatten (cycles):\t%.2f\n", (double)(flatten_end.QuadPart - flatten_start.QuadPart) / total_ticks);
        // printf("FC (cycles):\t\t%.2f\n\n", (double)(fc_end.QuadPart - fc_start.QuadPart) / total_ticks);
    }
    return pred;
}

int evaluate_from_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("ERROR: Unable to open file\n");
        return -1;
    }

    char line[4096];
    int total = 0;
    int correct = 0;
    // FOR PERFORMANCE TESTING ONLY SET maxCOUNT to 0 for default behaviour
    int maxCount = 1<<30; // Set custom number of samples to evaluate. Leave 0 if all.
    int counter = 1; // Used for above
    double total_cycles_conv2d = 0;
    double total_cycles_flatten = 0;
    double total_cycles_fc = 0;

    while (fgets(line, sizeof(line), file)) {
        
        // Skip empty lines
        if (strlen(line) < 10) continue;

        // Tokenize line
        char *token = strtok(line, ",");
        if (!token) continue;

        int label = atoi(token);
        float input[28][28];

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                token = strtok(NULL, ",");
                if (!token) {
                    printf("Invalid row: missing pixels!\n");
                    fclose(file);
                    return -1;
                }
                input[i][j] = atof(token) / 255.0f;
            }
        }
        // Predict number from this image
        float probs[FC_OUT];
        int prediction = predict(
            input,
            maxCount, 
            &total_cycles_conv2d, 
            &total_cycles_flatten, 
            &total_cycles_fc
        );
        if (prediction == label) correct++;
        total++;

        if (maxCount) {
            if (counter == maxCount) break;
            else counter++;
        }
    }
    fclose(file);
    float acc = 100.0f * correct / total;
    printf("Accuracy: %d/%d correct %.2f\n", correct, total, acc);

    // Print average cycles usage for each major operation
    if (maxCount) {
        double avg_conv2d = total_cycles_conv2d / total;
        double avg_flatten = total_cycles_flatten / total;
        double avg_fc = total_cycles_fc / total;
        // printf("%f\n%f\n%f\n\n", total_cycles_conv2d, total_cycles_flatten, total_cycles_fc);
        printf("Average Proportion of CPU time for each major operation:\n");
        printf("Conv2D:\t\t%.2f%\n", avg_conv2d*100);
        printf("Flatten:\t%.2f%\n", avg_flatten*100);
        printf("FC:\t\t%.2f%\n", avg_fc*100);
    }
}

int main() {
    load_conv_weights();
    load_fc_weights();

    evaluate_from_csv("../mnist/mnist_test.csv");
    // FC WEIGHTS CHECK
    // for (int k = 0; k < NUM_KERNELS; k++) {
    //     for (int i = 0; i < KERNEL_SIZE; i++) {
    //         for (int j = 0; j < KERNEL_SIZE; j++) {
    //             printf("%.4f ", conv_W[k][i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // FC BIASES CHECK
    // for (int i = 0; i < NUM_KERNELS; i++) {
    //     printf("%.4f ", conv_B[i]);
    // }
    // printf("\n");


    return 0;
}
