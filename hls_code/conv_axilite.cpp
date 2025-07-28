#include "conv_axilite.hpp"
#include "conv_biases.hpp"
#include "conv_weights.hpp"

static float conv_W[NUM_KERNELS][KERNEL_SIZE][KERNEL_SIZE];
static float conv_B[NUM_KERNELS];

// Initialisation routine
static void init_weights() {
#pragma HLS INLINE off
    int idx = 0;
    for (int k=0;k<NUM_KERNELS;k++)
        for (int i=0;i<KERNEL_SIZE;i++)
            for (int j=0;j<KERNEL_SIZE;j++)
                conv_W[k][i][j] = conv_weights[idx++];

    for (int k=0;k<NUM_KERNELS;k++)
        conv_B[k] = conv_biases[k];
}

static float relu(pixel_t x) {
#pragma HLS INLINE
	return x > 0 ? x : 0;
}

void cmac_unit_stream(pixel_t window[KERNEL_SIZE][KERNEL_SIZE], int kernel_id, pixel_t *acc) {
	// Place holder array for sums since I want an adder tree
	pixel_t partial_sum[KERNEL_SIZE * KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=partial_sum type=complete

	// Calculate the dot products in parallel
	int idx = 0;
	for (int i = 0; i < KERNEL_SIZE; i++) {
#pragma HLS UNROLL
		for (int j = 0; j < KERNEL_SIZE; j++) {
#pragma HLS UNROLL
			partial_sum[idx++] = window[i][j] * conv_W[kernel_id][i][j];
		}
	}

	/* NOTE: After some pain I realised why HLS won't make an adder tree if using floats...
	* Floating point addition is actually non-associative if the magnitudes differ greatly
	* So HLS/Compilers are conservative about trying to perform these operations out of order (e.g. adder tree)
	* BUT since the numbers we're dealing with are pretty close in magnitude this isn't really a problem
	* There IS a way to force allow via compiler config but this is too much work
	* I'd rather just manually construct this tree myself - Ken
	*/
	// Then accumulate via adder tree
	// MANUAL ADDER TREE
	// Level 1
	pixel_t sum1 = partial_sum[0] + partial_sum[1];
	pixel_t sum2 = partial_sum[2] + partial_sum[3];
	pixel_t sum3 = partial_sum[4] + partial_sum[5];
	pixel_t sum4 = partial_sum[6] + partial_sum[7];
	pixel_t sum5 = partial_sum[8] + conv_B[kernel_id];
	// Level 2
	pixel_t sum6 = sum1 + sum2;
	pixel_t sum7 = sum3 + sum4;
	// Level 3
	pixel_t sum8 = sum6 + sum7;
	// Level 4
	*acc = sum5 + sum8;
	// AUTOMATIC: USE ONLY IF NOT USING DOUBLE/FLOAT
//	pixel_t sum = 0.0f;
//	for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
//#pragma HLS UNROLL
//		sum += partial_sum[i];
//	}
//	*acc = sum + conv_B[kernel_id];
}

void conv_axilite(pixel_t input_image[IMG_SIZE][IMG_SIZE], feature_t output_feature_map[NUM_KERNELS][OUT_SIZE][OUT_SIZE]) {
	// Setup AXI-lite interface
#pragma HLS INTERFACE s_axilite port=input_image
#pragma HLS ARRAY_PARTITION variable=input_image type=complete
#pragma HLS INTERFACE s_axilite port=output_feature_map
#pragma HLS ARRAY_PARTITION variable=output_feature_map type=complete
#pragma HLS INTERFACE s_axilite port=return

	// Partition weights to be accessible in parallel
#pragma HLS ARRAY_PARTITION variable=conv_W type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_W type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv_W type=complete dim=3
#pragma HLS ARRAY_PARTITION variable=conv_B type=complete

	// Execute weight init on first launch only
	static bool weights_loaded = false;
	if (!weights_loaded) {
		init_weights();
		weights_loaded = true;
	}
	// Convolution HW design
	// 3-line buffer
	pixel_t linebuffer[3][IMG_SIZE];
#pragma HLS ARRAY_PARTITION variable=linebuffer type=complete dim=1
	// 3x3 image patch
	pixel_t window[KERNEL_SIZE][KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=window type=complete dim=0
	// Local output buffer to batch feature_map writes
	pixel_t local_fm[NUM_KERNELS][OUT_SIZE][OUT_SIZE];
#pragma HLS ARRAY_PARTITION variable=local_fm block factor=8 dim=1

	triline_buffer: for (int row = 0; row < IMG_SIZE; row++) {
		eachPatch_loop: for (int col = 0; col < IMG_SIZE; col++) {
#pragma HLS PIPELINE II=1
			// Progress line buffer;
//			pixel_t input_pixel = *input_image;
//			pixel_t input_pixel = input_image[row * IMG_SIZE + col];
			pixel_t input_pixel = input_image[row][col];
			linebuffer[0][col] = linebuffer[1][col];
			linebuffer[1][col] = linebuffer[2][col];
			linebuffer[2][col] = input_pixel;

			patchExtract_loop: if (row >= 2 && col >= 2) {
				for (int i = 0; i < KERNEL_SIZE; i++) {
#pragma HLS UNROLL
					for (int j = 0; j < KERNEL_SIZE; j++) {
#pragma HLS UNROLL
						window[i][j] = (i < 2) ? linebuffer[i][col - 2 + j] : linebuffer[2][col - 2 + j];
					}
				}
				pixel_t sum[NUM_KERNELS];
#pragma HLS ARRAY_PARTITION variable=sum type=complete
				// For each kernel (CMAC unit)
				cmac_loop: for (int k = 0; k < NUM_KERNELS; k++) {
#pragma HLS UNROLL
					pixel_t acc = 0.0f;
					// Inside each CMAC unit
					// Calculate the dot products in parallel and accumulate via adder tree
					cmac_unit_stream(window, k, &acc);
					sum[k] = relu(acc); // reLU unit
				}
				// Then place the sums in correct indexes
				int out_row = row - 2;
				int out_col = col - 2;
				write_to_localfm_loop: for (int k = 0; k < NUM_KERNELS; k++) {
#pragma HLS UNROLL
					local_fm[k][out_row][out_col] = sum[k];
				}
			}
		}
	}
	// Once the feature map finally computed,
	// upload local feature map buffer to the external memory feature map (via AXI)
	writeFM_serially_loop: for (int k = 0; k < NUM_KERNELS; k++) {
#pragma HLS UNROLL // 8 writers
		for (int i = 0; i < OUT_SIZE; i++) {
			for (int j = 0; j < OUT_SIZE; j++) {
#pragma HLS PIPELINE II=1
//				output_feature_map[k * OUT_SIZE * OUT_SIZE + i * OUT_SIZE + j] = local_fm[k][i][j];
//				*output_feature_map = local_fm[k][i][j];
				output_feature_map[k][i][j] = local_fm[k][i][j];
			}
		}
	}
}
