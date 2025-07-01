import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import random

def save_random_mnist_entry_as_header_from_test_data(test_data, output_path):
    """
    test_data: numpy array or DataFrame, shape (num_samples, 785),
               with first column label, rest 784 pixels.
    output_path: path to write the C header file.
    """

    # Pick random sample index within test_data
    sample_index = random.randint(0, len(test_data) - 1)
    row = test_data[sample_index]

    label = int(row[0])
    image = row[1:].reshape(28, 28) / 255.0  # Normalize pixels

    # Write to header file
    with open(output_path, 'w') as f:
        f.write('#ifndef TEST_IMAGE_DATA_H\n')
        f.write('#define TEST_IMAGE_DATA_H\n\n')
        f.write('#define IMG_SIZE 28\n\n')
        f.write('static const float test_image[IMG_SIZE][IMG_SIZE] = {\n')
        for i in range(28):
            row_str = ', '.join(f'{pixel:.6f}' for pixel in image[i])
            f.write(f'    {{{row_str}}}')
            f.write(',\n' if i < 27 else '\n')
        f.write('};\n\n')
        f.write(f'static const int test_label = {label};\n\n')
        f.write('#endif // TEST_IMAGE_DATA_H\n')

    # Display chosen test image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label} (test index {sample_index})')
    plt.axis('off')
    plt.show()

def export_model_to_c(filename="cnn_model.h"):
    with open(filename, "w") as f:
        num_kernels = len(kernels)
        kernel_size = len(kernels[0])
        output_classes = len(W_dense)
        dense_input_size = len(W_dense[0])

        f.write("#ifndef CNN_MODEL_H\n#define CNN_MODEL_H\n\n")
        f.write(f"#define NUM_KERNELS {num_kernels}\n")
        f.write(f"#define KERNEL_SIZE {kernel_size}\n")
        f.write(f"#define DENSE_OUT {output_classes}\n")
        f.write(f"#define DENSE_IN {dense_input_size}\n\n")

        # Write kernels array
        f.write(f"static const float kernels[NUM_KERNELS][KERNEL_SIZE][KERNEL_SIZE] = {{\n")
        for k in kernels:
            f.write("  {\n")
            for row in k:
                row_str = ", ".join(f"{val:.6f}" for val in row)
                f.write(f"    {{{row_str}}},\n")
            f.write("  },\n")
        f.write("};\n\n")

        # Write dense biases
        bias_str = ", ".join(f"{val.item():.6f}" for val in np.array(b_dense).flatten())
        f.write(f"static const float b_dense[DENSE_OUT] = {{{bias_str}}};\n\n")

        # Write dense weights
        f.write(f"static const float W_dense[DENSE_OUT][DENSE_IN] = {{\n")
        for row in W_dense:
            row_str = ", ".join(f"{val:.6f}" for val in row)
            f.write(f"  {{{row_str}}},\n")
        f.write("};\n\n")

        f.write("#endif // CNN_MODEL_H\n")

    print(f"✅ Model exported to {filename}")

# Activations
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return x > 0
def softmax(x): 
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=0)

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, output_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T  # shape (10, m)

# Forward pass
def forward(image):
    feature_maps = [relu(convolve2d(image, k, mode='valid') + b) for k, b in zip(kernels, biases)]
    stacked = np.array(feature_maps)
    flat = stacked.reshape(-1, 1)  # shape (dense_input_size, 1)
    Z = W_dense @ flat + b_dense
    A = softmax(Z)
    return feature_maps, flat, Z, A

# Backward pass
def backward(flat, Z, A, Y_true):
    one_hot_Y = one_hot(np.array([Y_true]))
    dZ = A - one_hot_Y  # shape (10, 1)
    dW = dZ @ flat.T
    db = dZ
    return dW, db

# Training loop
def train_dense(X_train, Y_train, epochs=20, lr=0.01):
    global W_dense, b_dense
    for epoch in range(epochs):
        correct = 0
        for i in range(len(X_train)):
            image = X_train[i]
            label = Y_train[i]

            # Forward
            fmaps, flat, Z, A = forward(image)

            # Prediction
            pred = np.argmax(A)
            if pred == label:
                correct += 1

            # Backward
            dW, db = backward(flat, Z, A, label)

            # Update
            W_dense -= lr * dW
            b_dense -= lr * db

        acc = correct / len(X_train)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")


# Load and preprocess MNIST
csv_path = 'mnist_test.csv'
header_path = 'test_image_data.h'

data = pd.read_csv(csv_path)
data = np.array(data)
np.random.shuffle(data)
# Split training and testing data
X = data[:, 1:].reshape(-1, 28, 28) / 255.0
Y = data[:, 0]

X_train, Y_train = X[:1500], Y[:1500]  # use more data if possible
test_data = data[1501:]



# CNN params
num_kernels = 8
kernel_size = 3
output_classes = 10

# Initialize conv kernels (fixed)
kernels = [np.random.randn(kernel_size, kernel_size) * 0.1 for _ in range(num_kernels)]
biases = [0.0 for _ in range(num_kernels)]

# Dense layer setup
conv_output_size = 26  # 28 - 3 + 1
dense_input_size = conv_output_size * conv_output_size * num_kernels

W_dense = np.random.randn(output_classes, dense_input_size) * np.sqrt(2 / dense_input_size)
b_dense = np.zeros((output_classes, 1))

# Run training
train_dense(X_train, Y_train, epochs=20, lr=0.01)
export_model_to_c("cnn_model.h")

# Save a random test sample from the last 1000 entries to C header file
save_random_mnist_entry_as_header_from_test_data(test_data, header_path)




