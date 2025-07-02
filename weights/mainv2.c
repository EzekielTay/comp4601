#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "conv_weights.h"
#include "conv_biases.h"
#include "fc_weights.h"
#include "fc_biases.h"

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
int predict(float image[IMG_SIZE][IMG_SIZE]) {
    float conv_out[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
    float flat[NUM_KERNELS * OUT_SIZE * OUT_SIZE];
    float logits[FC_OUT];

    conv2d(image, conv_out);
    flatten(conv_out, flat);
    fc(flat, logits);

    // Argmax
    int pred = 0;
    float max_val = logits[0];
    for (int i = 1; i < FC_OUT; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            pred = i;
        }
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
        int prediction = predict(input);
        if (prediction == label) correct++;
        total++;
    }
    fclose(file);
    float acc = 100.0f * correct / total;
    printf("Accuracy: %d/%d correct %.2f\n", correct, total, acc);
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
