1D-CNN Model for Response Matrix Analysis at Indus-2
This repository contains the implementation of a 1D Convolutional Neural Network (CNN) developed during my internship at Raja Ramanna Centre for Advanced Technology (RRCAT), Indore. The model analyzes response matrices of steering magnets used in the Indus-2 facility, aiming to enhance precision and control in beam steering.

Project Overview
Objective
To design and implement a 1D-CNN model that processes horizontal and vertical beam positions and predicts the activation states of steering magnets in the Indus-2 facility.

Key Features
Input Channels: Processes two channels for horizontal and vertical beam positions.
Circular Padding: Incorporates circular padding to reduce errors, leveraging the closed-loop nature of steering magnet positions.
Output Nodes:
88 nodes representing individual steering magnets.
1 additional node to detect inactive states.
Dataset
The dataset comprises response matrices determined and used at the Indus-2 facility:

Training Data: Three response matrices.
Testing Data: One response matrix.
Model Architecture
The CNN model consists of:

Input Layer: Accepts a 2D input array (horizontal and vertical beam positions).
Convolutional Layers: Utilizes different kernels to process the input channels.
Output Layer:
Outputs a probability distribution over 89 classes (88 magnets + 1 inactive state).
