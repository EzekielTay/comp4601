import numpy as np
from scipy.signal import convolve2d
import re
# === Load CNN model from C header ===
def load_cnn_model(filename="cnn_model.h"):
    with open(filename, "r") as f:
        content = f.read()

    # === Parse kernels ===
    kernel_start = content.find("static const float kernels")
    if kernel_start == -1:
        raise ValueError("kernels block not found")

    kernel_block = content[kernel_start:]
    kernel_data_start = kernel_block.find("{")
    kernel_data_end = kernel_block.find("};") + 1
    kernel_raw = kernel_block[kernel_data_start:kernel_data_end]

    # Clean and parse nested array
    kernel_raw = kernel_raw.replace("\n", "").replace(" ", "")
    kernel_raw = kernel_raw.replace("{", "[").replace("}", "]")
    kernels = eval(kernel_raw)  # Safe only if file is trusted!
    kernels = [np.array(k) for k in kernels]

    # === Parse dense biases ===
    bias_start = content.find("static const float b_dense")
    if bias_start == -1:
        raise ValueError("b_dense block not found")

    bias_block = content[bias_start:]
    bias_data_start = bias_block.find("{")
    bias_data_end = bias_block.find("};")
    bias_raw = bias_block[bias_data_start + 1 : bias_data_end]
    b_dense = np.array([[float(x)] for x in bias_raw.split(",") if x.strip()])

    # === Parse dense weights ===
    W_start = content.find("static const float W_dense")
    if W_start == -1:
        raise ValueError("W_dense block not found")

    W_block = content[W_start:]
    W_data_start = W_block.find("{")
    W_data_end = W_block.find("};")
    W_lines = W_block[W_data_start + 1 : W_data_end].split("},")
    dense_weights = []
    for line in W_lines:
        line = line.strip().lstrip("{").rstrip("}")
        nums = [float(x.strip()) for x in line.split(",") if x.strip()]
        if nums:
            dense_weights.append(nums)

    W_dense = np.array(dense_weights)

    return kernels, W_dense, b_dense


# === Activation functions ===

def relu(x): return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=0)

# === Inference ===
def predict(image, kernels, W_dense, b_dense):
    feature_maps = [relu(convolve2d(image, k, mode='valid')) for k in kernels]
    stacked = np.array(feature_maps)  # shape: (num_kernels, 26, 26)
    flat = stacked.reshape(-1, 1)
    Z = W_dense @ flat + b_dense
    A = softmax(Z)
    return np.argmax(A), A


def parse_header_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Extract the label
    label_line = [line for line in lines if 'static const int test_label' in line][0]
    label = int(re.search(r'\d+', label_line).group())

    # Extract the image array
    array_lines = []
    in_array = False
    for line in lines:
        if 'static const float test_image' in line:
            in_array = True
            continue
        if in_array:
            if '};' in line:
                break
            array_lines.append(line.strip().strip(','))
    
    # Parse rows
    image = []
    for row in array_lines:
        numbers = [float(num) for num in row.strip('{}').split(',')]
        image.append(numbers)
    
    return np.array(image, dtype=np.float32), label


# === Test ===

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import pandas as pd

    # Load model
    kernels, W_dense, b_dense = load_cnn_model("cnn_model.h")

    # Load image and true label from C header
    image, label = parse_header_file("test_image_data.h")

    # Reshape image if needed
    image = image.reshape((28, 28))

    # Predict using the model
    predicted_label, probs = predict(image, kernels, W_dense, b_dense)

    # Print predicted, true label, and probabilities
    print(f"Predicted: {predicted_label}, True label: {label}")
    print("Probabilities per digit:")
    for digit, prob in enumerate(probs.ravel()):
        print(f"Digit {digit}: {prob:.6f}")

    # Show the image
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="gray")
    plt.title(f"Label from header: {label}")
    plt.show()
