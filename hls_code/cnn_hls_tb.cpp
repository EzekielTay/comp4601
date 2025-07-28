#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include "conv_axilite.hpp"
#include "conv_weights.hpp"
#include "conv_biases.hpp"
#include "fc_weights.hpp"
#include "fc_biases.hpp"

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

const static float input_7[784] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
;

int argmax(const float logits[], int length) {
    int best_i = 0;
    float best_v = logits[0];
    for (int i = 1; i < length; ++i) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

std::uint64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
               steady_clock::now().time_since_epoch()
           ).count();
}

int main() {
	std::cout << "Starting testbench for conv engine AXI-lite version" << std::endl;
	// Initialise weights
	load_conv_weights();
	load_fc_weights();
	// Open MNIST test csv file
	char filename[] = "./mnist_test.csv";
	FILE *file = fopen(filename, "r");
	if (!file) {
		printf("ERROR: Unable to open file\n");
		return -1;
	}

	char line[4096];
	int total = 0;
	int correct = 0;
	int maxCount = 0; // Custom number of samples to evaluate (set to 0 if do all)
	int counter = 1;
	auto time_start = now_ms();
	while (fgets(line, sizeof(line), file)) {
		// Skip empty lines
		if (strlen(line) < 10) continue;

		// Tokenize line
		char *token = strtok(line, ",");
		if (!token) continue;

		// Get correct label
		int label = atoi(token);
//		float input[28][28];
		pixel_t img[IMG_SIZE][IMG_SIZE];
		// Load image into array
		for (int i = 0; i < IMG_SIZE; i++) {
			for (int j = 0; j < IMG_SIZE; j++) {
				token = strtok(NULL, ",");
				if (!token) {
					printf("Invalid row: missing pixels!\n");
					fclose(file);
					return -1;
				}
				img[i][j] = atof(token) / 255.0f;
			}
		}

		// Predict number from this image via CNN
		float conv_out[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
		float flat[NUM_KERNELS * OUT_SIZE * OUT_SIZE];
		float logits[FC_OUT];
		// 2D convolution
//		conv2d(input, conv_out); // PS-side implementation (Baseline)
		// PL-side accelerator core NON-STREAM

		feature_t output_fm[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
//		conv_axilite(&img[0][0], &output_fm[0][0][0]);
		conv_axilite(img, output_fm);

		// Flatten
		flatten(output_fm, flat);
		// FC
		fc(flat, logits);
		// Argmax
		int pred = argmax(logits, 10);
		if (pred == label) correct++;
		total++;

		if (maxCount) {
			if (counter == maxCount) break;
			else counter++;
		}
	}
	auto time_end = now_ms();
	std::cout << "Time elapsed: " << (time_end - time_start) << " ms\n";
	fclose(file);
	float acc = 100.0f * correct / total;
	printf("Accuracy: %d/%d correct %.2f\n", correct, total, acc);

    return 0;
}
