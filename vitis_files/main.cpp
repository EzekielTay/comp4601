#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdint>
#include <chrono>

#include "conv_weights.hpp"
#include "conv_biases.hpp"
#include "fc_weights.hpp"
#include "fc_biases.hpp"
#include "conv_axilite.hpp"

#define USER_IP_ADDR_LOW 0xB0010000
#define USER_IP_ADDR_HIGH 0xB001FFFF

#define USER_IP_ADDR_OFFSET_CTRL 0
#define AP_START_BIT 0
#define AP_DONE_BIT 1
#define AP_IDLE_BIT 2
#define AP_READY_BIT 3

#define USER_IP_ADDR_OFFSET_RESULT 0x218
#define USER_IP_ADDR_OFFSET_RESULT_VLD 0x21C
#define USER_IP_ADDR_OFFSET_INPUT 0x10
#define USER_IP_ADDR_OFFSET_FILTER_0 0x18

#define INPUT_IMAGE_OFFSET_BASE 0x80
#define INPUT_IMAGE_BLOCK_SIZE 0x80

#define OUTPUT_FM_OFFSET_BASE 0x1000
#define OUTPUT_FM_BLOCK_SIZE 0x1000

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
void flatten(feature_t input[NUM_KERNELS][OUT_SIZE][OUT_SIZE], feature_t output[NUM_KERNELS * OUT_SIZE * OUT_SIZE]) {
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

void initialize_hw(void) {
	if(getuid() != 0) {
		printf("reconfiguration flow requires super user!\n");
		return;
	}
	// configure FPGA & device tree
	system("sudo xmutil unloadapp"); // remove existing dt overlay
	system("sudo xmutil loadapp my_app"); // program FPGA and load dt overlay
}

// The "Unoptimised" version of toyCNN
void run_localCNN() {
	load_fc_weights();
	load_conv_weights();
	printf("\n\nRunning Unoptimised toyCNN (Entirely PS-side)\n");
	// Check if dataset is there
	char cwd[256];
	if (getcwd(cwd, sizeof(cwd)))
		printf("CWD on target: %s\n", cwd);
	else
		perror("getcwd");
	// Open image dataset file
	char filename[] = "./mnist_test.csv";
	FILE *file = fopen(filename, "r");
	if (!file) {
		printf("ERROR: Unable to open file\n");
	}
	printf("Succesfully opened %s.\n", filename);

	char line[4096];
	int total = 0;
	int correct = 0;
	int maxCount = 10000; // Custom number of smaples to evaluate (set to 0 if do all)
	int counter = 1;
	auto time_start = now_ms();
	// Per-stage running average percentages
	double avg_load_img_pct     = 0.0;
	double avg_conv_pct        	= 0.0;
	double avg_flatten_pct   	= 0.0;
	double avg_fc_pct        	= 0.0;

	printf("Running %d images through the toyCNN, with the CONV layer computed in the PS.\n", maxCount);
	while (fgets(line, sizeof(line), file)) {
		// Skip empty lines
		if (strlen(line) < 10) continue;

		// Tokenize line
		char *token = strtok(line, ",");
		if (!token) continue;

		// Get correct label
		int label = atoi(token);
		float input[IMG_SIZE][IMG_SIZE];
		// Load image into local buffer
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < IMG_SIZE; i++) {
			// Load pixels row into buffer
			for (int j = 0; j < IMG_SIZE; j++) {
				token = strtok(NULL, ",");
				if (!token) {
					printf("Invalid row: missing pixels!\n");
					fclose(file);
				}
				float pixel_val = atof(token) / 255.0f;
				input[i][j] = pixel_val;
			}
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		// Perform PS-side convolution
		float output_fm[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
		conv2d(input, output_fm);
		auto t2 = std::chrono::high_resolution_clock::now();

		// Perform remaining layers on PS
		float flat[NUM_KERNELS * OUT_SIZE * OUT_SIZE];
		float logits[FC_OUT];

		auto t4_start = std::chrono::high_resolution_clock::now();
		flatten(output_fm, flat);
		auto t4_end = std::chrono::high_resolution_clock::now();

		auto t5_start = std::chrono::high_resolution_clock::now();
		fc(flat, logits);
		auto t5_end = std::chrono::high_resolution_clock::now();

		int pred = argmax(logits, 10);
//		printf("Pred: %d Label: %d\n", pred, label);
		if (pred == label) correct++;
		total++;

		// Get all layer durations
		auto d_load_img = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		auto d_conv     = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		auto d_flatten 	= std::chrono::duration_cast<std::chrono::nanoseconds>(t4_end - t4_start).count();
		auto d_fc      	= std::chrono::duration_cast<std::chrono::nanoseconds>(t5_end - t5_start).count();
		// Compute percentages that each step took
		long long total_ns 	= d_load_img + d_conv + d_flatten + d_fc;
		double pct_load_img = 100.0 * d_load_img    / total_ns;
		double pct_conv     = 100.0 * d_conv      	/ total_ns;
		double pct_flatten 	= 100.0 * d_flatten 	/ total_ns;
		double pct_fc      	= 100.0 * d_fc      	/ total_ns;
		// Update running averages
		double inv_count 	= 1.0 / total;
		avg_load_img_pct	+= (pct_load_img    - avg_load_img_pct)	* inv_count;
		avg_conv_pct    	+= (pct_conv      	- avg_conv_pct)    	* inv_count;
		avg_flatten_pct 	+= (pct_flatten 	- avg_flatten_pct) 	* inv_count;
		avg_fc_pct      	+= (pct_fc      	- avg_fc_pct)      	* inv_count;


		if (maxCount) {
			if (counter == maxCount) break;
			else counter++;
		}
	}
	auto time_end = now_ms();
	printf("Batch inference complete!\n");
	std::cout << "Time elapsed: " << (time_end - time_start) << " ms\n";
	fclose(file);
	float acc = 100.0f * correct / total;
	printf("Accuracy: %d/%d correct %.2f\n", correct, total, acc);
	printf("-------------Running Averages of each major operation-------------\n");
	printf("Loading image into memory:\t\t%.2f%%\n", avg_load_img_pct);
	printf("Convolution layer on the PS:\t\t%.2f%%\n", avg_conv_pct);
	printf("Flatten layer:\t\t\t\t%.2f%%\n", avg_flatten_pct);
	printf("FC layer:\t\t\t\t%.2f%%\n", avg_fc_pct);
	printf("------------------------------------------------------------------\n");
}

int main(void) {
	printf("Initialising HW...\n");
	initialize_hw();
	printf("HW initialised.\n");

	printf("Opening /dev/mem\n");
	// bit hacky, should be properly mapped as userspace i/o instead
	auto fd = open("/dev/mem", O_RDWR|O_SYNC);
	if (fd == -1) {
		perror("/dev/mem open");
		return 0;
	}
	printf("Mmapping axi base pointer.\n");
	uint64_t axiBasePtr = (uint64_t)mmap(NULL, USER_IP_ADDR_HIGH-USER_IP_ADDR_LOW, PROT_READ|PROT_WRITE, MAP_SHARED, fd, USER_IP_ADDR_LOW);
	close(fd); // fd can be closed after mmap is created
	if ((void*)axiBasePtr == MAP_FAILED)
	{
	   perror("Error mapping the device to memory.\n");
	   exit(EXIT_FAILURE);
	}
	printf("Setup complete.\n");

	printf("Wait for IP Ready.\n");
	while (!(*(uint32_t*)(axiBasePtr+USER_IP_ADDR_OFFSET_CTRL) & (1 << AP_IDLE_BIT))) {
		sleep(2);
		printf("IP CTRL Flag: %d\n", *(uint32_t*)(axiBasePtr+USER_IP_ADDR_OFFSET_CTRL));
	}
	printf("IP Ready!\n");

	printf("SETTING UP MNIST TEST\n");
	load_fc_weights();
	// Check if dataset is there
	char cwd[256];
	if (getcwd(cwd, sizeof(cwd)))
		printf("CWD on target: %s\n", cwd);
	else
		perror("getcwd");
	// Open image dataset file
	char filename[] = "./mnist_test.csv";
	FILE *file = fopen(filename, "r");
	if (!file) {
		printf("ERROR: Unable to open file\n");
		return -1;
	}
	printf("Succesfully opened %s.\n", filename);

	char line[4096];
	int total = 0;
	int correct = 0;
	int maxCount = 10000; // Custom number of samples to evaluate (set to 0 if do all)
	int counter = 1;
	auto time_start = now_ms();
	// Per-stage running average percentages
	double avg_load_img_pct     = 0.0;
	double avg_conv_pct        	= 0.0;
	double avg_flatten_pct   	= 0.0;
	double avg_fc_pct        	= 0.0;

	printf("Running %d images through the toyCNN, with the CONV layer accelerated by the PL.\n", maxCount);
	while (fgets(line, sizeof(line), file)) {
		// Skip empty lines
		if (strlen(line) < 10) continue;

		// Tokenize line
		char *token = strtok(line, ",");
		if (!token) continue;

		// Get correct label
		int label = atoi(token);
		float input[IMG_SIZE][IMG_SIZE];
		// Load image into local buffer then send to shared MM
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < IMG_SIZE; i++) {
			// Load pixels row into buffer
			for (int j = 0; j < IMG_SIZE; j++) {
				token = strtok(NULL, ",");
				if (!token) {
					printf("Invalid row: missing pixels!\n");
					fclose(file);
					return -1;
				}
				float pixel_val = atof(token) / 255.0f;
				input[i][j] = pixel_val;
			}
			// Then mem copy to the shared MM
			uint32_t *rowPtr = (uint32_t*)(axiBasePtr + INPUT_IMAGE_OFFSET_BASE + INPUT_IMAGE_BLOCK_SIZE * i);
			memcpy(rowPtr, input[i], sizeof(pixel_t) * IMG_SIZE);
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		// Start IP
		*(uint32_t*)(axiBasePtr+USER_IP_ADDR_OFFSET_CTRL) = 1 << AP_START_BIT;
//		printf("IP CTRL Flag: %d\n", *(uint32_t*)(axiBasePtr+USER_IP_ADDR_OFFSET_CTRL));
//		printf("Waiting for conv IP to finish...\n");
		while(!(*(uint32_t*)(axiBasePtr+USER_IP_ADDR_OFFSET_CTRL) & 0b110)) {
		}
		auto t2 = std::chrono::high_resolution_clock::now();
//		printf("conv IP finished!\n");
		// Perform remaining layers on PS
		float flat[NUM_KERNELS * OUT_SIZE * OUT_SIZE];
		float logits[FC_OUT];
		
		// Flatten layer
		auto t4_start = std::chrono::high_resolution_clock::now();
		// flatten(output_fm, flat);
		for (int k = 0; k < NUM_KERNELS; k++) {
			uint32_t *kernelPtr = (uint32_t*)(axiBasePtr + OUTPUT_FM_OFFSET_BASE + OUTPUT_FM_BLOCK_SIZE*k);
			memcpy(&flat[k*OUT_SIZE*OUT_SIZE], kernelPtr, sizeof(float) * OUT_SIZE * OUT_SIZE);
		}
		auto t4_end = std::chrono::high_resolution_clock::now();

		auto t5_start = std::chrono::high_resolution_clock::now();
		fc(flat, logits);
		auto t5_end = std::chrono::high_resolution_clock::now();

		int pred = argmax(logits, 10);
//		printf("Pred: %d Label: %d\n", pred, label);
		if (pred == label) correct++;
		total++;

		// Get all layer durations
		auto d_load_img = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		auto d_conv     = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		auto d_flatten 	= std::chrono::duration_cast<std::chrono::nanoseconds>(t4_end - t4_start).count();
		auto d_fc      	= std::chrono::duration_cast<std::chrono::nanoseconds>(t5_end - t5_start).count();
		// Compute percentages that each step took
		long long total_ns 	= d_load_img + d_conv + d_flatten + d_fc;
		double pct_load_img = 100.0 * d_load_img    / total_ns;
		double pct_conv     = 100.0 * d_conv      	/ total_ns;
		double pct_flatten 	= 100.0 * d_flatten 	/ total_ns;
		double pct_fc      	= 100.0 * d_fc      	/ total_ns;
		// Update running averages
		double inv_count 	= 1.0 / total;
		avg_load_img_pct	+= (pct_load_img    - avg_load_img_pct) * inv_count;
		avg_conv_pct    	+= (pct_conv      	- avg_conv_pct)     * inv_count;
		avg_flatten_pct 	+= (pct_flatten 	- avg_flatten_pct) 	* inv_count;
		avg_fc_pct      	+= (pct_fc      	- avg_fc_pct)      	* inv_count;

		if (maxCount) {
			if (counter == maxCount) break;
			else counter++;
		}
	}
	auto time_end = now_ms();
	printf("Batch inference complete!\n");
	std::cout << "Time elapsed: " << (time_end - time_start) << " ms\n";
	fclose(file);
	float acc = 100.0f * correct / total;
	printf("Accuracy: %d/%d correct %.2f\n", correct, total, acc);
	printf("-------------Running Averages of each major operation-------------\n");
	printf("Loading image into shared memory:\t%.2f%%\n", avg_load_img_pct);
	printf("Convolution layer on the PL:\t\t%.2f%%\n", avg_conv_pct);
	printf("FM in shared mem to flat layer:\t\t%.2f%%\n", avg_flatten_pct);
	printf("FC layer:\t\t\t\t%.2f%%\n", avg_fc_pct);
	printf("------------------------------------------------------------------\n");
	// Now running an entirely PS-side CNN version for comparison
	run_localCNN();

	printf("Test Complete!\n");
	return 0;
}
