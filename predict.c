#include "cnn_model.h"
#include "test_image_data.h"
#include <stdio.h>
#include <math.h> 

#define KERNEL_SIZE 3
#define NUM_KERNELS 8
#define DENSE_OUT 10
#define CONV_SIZE 26  // (IMG_SIZE - KERNEL_SIZE + 1)
#define IMG_SIZE 28
#define MAX_SIZE 128
#define DENSE_IN 5408  // (NUM_KERNELS * CONV_SIZE * CONV_SIZE)

void relu(float feature[CONV_SIZE][CONV_SIZE]);
void softmax(float x[DENSE_OUT]);
void convolve2d(const float image[IMG_SIZE][IMG_SIZE], const float kernel[KERNEL_SIZE][KERNEL_SIZE], float output[CONV_SIZE][CONV_SIZE]);
void flatten(float feature_maps[NUM_KERNELS][CONV_SIZE][CONV_SIZE], float flat[DENSE_IN]);
void dense_layer(const float W_dense[DENSE_OUT][DENSE_IN], const float flat[DENSE_IN], const float b_dense[DENSE_OUT], float result[DENSE_OUT]);
int predict(
    const float image[IMG_SIZE][IMG_SIZE],
    const float kernels[NUM_KERNELS][KERNEL_SIZE][KERNEL_SIZE],
    const float W_dense[DENSE_OUT][DENSE_IN],
    const float b_dense[DENSE_OUT],
    float probs[DENSE_OUT]   // output softmax probabilities
);


// Activation function on the 26 x 26 feature map produced from convolution layer
void relu(float feature[CONV_SIZE][CONV_SIZE]) {
    for (int i = 0; i < CONV_SIZE; i++) {
        for (int j = 0; j < CONV_SIZE; j++) {
            if (feature[i][j] < 0) {
                feature[i][j] = 0;
            }
        }
    }
}

void softmax(float x[DENSE_OUT]) {
    float max_val = x[0];
    for (int i = 1; i < DENSE_OUT; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    float exp_vals[DENSE_OUT];

    for (int i = 0; i < DENSE_OUT; i++) {
        exp_vals[i] = expf(x[i] - max_val);
        sum += exp_vals[i];
    }

    for (int i = 0; i < DENSE_OUT; i++) {
        x[i] = exp_vals[i] / sum;
    }
}


// === 2D valid convolution ===
// image: [28][28]
// kernel: [3][3]
// output: [26][26]
void convolve2d(const float image[IMG_SIZE][IMG_SIZE], const float kernel[KERNEL_SIZE][KERNEL_SIZE], float output[CONV_SIZE][CONV_SIZE]) {
    for (int i = 0; i < CONV_SIZE; i++) {
        for (int j = 0; j < CONV_SIZE; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    sum += image[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
}

// === Flatten feature maps ===
// input: feature_maps[num_kernels][26][26]
// output: flat[DENSE_IN]
void flatten(float feature_maps[NUM_KERNELS][CONV_SIZE][CONV_SIZE], float flat[DENSE_IN]) {
    int index = 0;
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int i = 0; i < CONV_SIZE; i++) {
            for (int j = 0; j < CONV_SIZE; j++) {
                flat[index++] = feature_maps[k][i][j];
            }
        }
    }
}

// === Dense layer: Z = W * flat + b ===
// W_dense: [DENSE_OUT][DENSE_IN]
// flat: [DENSE_IN]
// b_dense: [DENSE_OUT]
// output: Z[DENSE_OUT]
void dense_layer(const float W_dense[DENSE_OUT][DENSE_IN], const float flat[DENSE_IN], const float b_dense[DENSE_OUT], float result[DENSE_OUT]) {
    for (int i = 0; i < DENSE_OUT; i++) {
        float sum = b_dense[i];
        for (int j = 0; j < DENSE_IN; j++) {
            sum += W_dense[i][j] * flat[j];
        }
        result[i] = sum;
    }
}

// === Predict function ===
// Returns index of max softmax prob, and fills probs output array
int predict(
    const float image[IMG_SIZE][IMG_SIZE],
    const float kernels[NUM_KERNELS][KERNEL_SIZE][KERNEL_SIZE],
    const float W_dense[DENSE_OUT][DENSE_IN],
    const float b_dense[DENSE_OUT],
    float probs[DENSE_OUT]   // output softmax probabilities
) {
    // 1) Convolve + relu for each kernel
    static float feature_maps[NUM_KERNELS][CONV_SIZE][CONV_SIZE];
    for (int k = 0; k < NUM_KERNELS; k++) {
        convolve2d(image, kernels[k], feature_maps[k]);
        relu(feature_maps[k]);
    }

    // 2) Flatten feature maps
    static float flat[DENSE_IN];
    flatten(feature_maps, flat);

    // 3) Dense layer
    static float Z[DENSE_OUT];
    dense_layer(W_dense, flat, b_dense, Z);

    // 4) Softmax
    for (int i = 0; i < DENSE_OUT; i++) probs[i] = Z[i];
    softmax(probs);
    // 5) Find max prob class
    int max_idx = 0;
    float max_val = probs[0];
    for (int i = 1; i < DENSE_OUT; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_idx = i;
        }
    }

    return max_idx;
}

int main() {
    float probs[DENSE_OUT];
    int predicted = predict(test_image, kernels, W_dense, b_dense, probs);

    printf("Predicted label: %d\n", predicted);
    printf("Actual label:    %d\n", test_label);

    printf("Softmax probabilities:\n");
    for (int i = 0; i < DENSE_OUT; i++) {
        printf("Class %d: %.5f\n", i, probs[i]);
    }
    return 0;
}

