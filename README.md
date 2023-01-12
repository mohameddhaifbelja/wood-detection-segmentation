# wood-detection-segmentation
Under a freelance agreement: This project contain the code source to develop an efficient system for wood lug detection and segmentation

# Code:

The folder python_code contains the scripts to run training using python and pytorch-lightning.

the subfolder code_to_be_converted contains the training and inference functions using pytroch which should be easy to integrated using C++. The main block contains example for running trainining and inference. In the inference function, I load the model everytime but when you implement the code in C++ you will need to load once.

Requirements.txt contains the packages that I needed to install in order to perfrom the training.
