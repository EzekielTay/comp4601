# COMP Neural Networks (CNN)
## Video
[Demo is available on YouTube](https://youtu.be/a3USThQnu3Y?si=RIt50kUofjgmsBfc)

## File structure
### trainer_code/
- requirements.txt
  - Python packages for trainer.ipynb
- trainer.ipynb
  - The Jupyter Notebook to build, train and test a CNN model.
- weights/main.c
  - The C code software implementation of the CNN (this version was designed to be run on Windows)
- weights/*biases.h and weights/*weights.h
  - Header files containing the biases and weights of the CNN
- mnist/
  - Contains the MNIST testing dataset in CSV format.
  - NOTE: The MNIST training dataset is not uploaded to the repo due to file size limits. (It contains 60k samples totalling to 107MB). You can download the csv from [here](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/#mnist-in-csv)
  - The CSV format is in the form:
    - label, pix-11, pix-12, pix-13, ...

### my_app/
Contains the device tree overlay (.dtbo) and HW binary (.bit.bin) to upload to the board.

### hls_code/
Contains the HLS code of the CNN's HW accelerator core. `conv_axilite.cpp`'s contains the top level function called **conv_axilite**. `cnn_hls_tb.cpp` is the testbench to test this top-level function. NOTE: You will need the MNIST test csv included in the HLS project for the testbench to read from.

### vitis_files/
This directory contains the PS-side software to be imported into the Vitis project (the fir axilite demo), with `main.cpp` replacing the original. The `*.hpp` files provide the weights and biases for the SW-version of the CNN for performance comparison.