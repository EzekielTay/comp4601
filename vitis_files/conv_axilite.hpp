#define IMG_SIZE 28
#define KERNEL_SIZE 3
#define NUM_KERNELS 8
#define STRIDE 1
#define OUT_SIZE (IMG_SIZE - KERNEL_SIZE + 1)
#define FC_IN (NUM_KERNELS * OUT_SIZE * OUT_SIZE)
#define FC_OUT 10

typedef float pixel_t; // Data type of the pixels
typedef float feature_t; // Data type of the feature map elements

//void conv_axilite(pixel_t *input, feature_t *output_feature_map);
void conv_axilite(pixel_t input_image[IMG_SIZE][IMG_SIZE], feature_t output_feature_map[NUM_KERNELS][OUT_SIZE][OUT_SIZE]);
