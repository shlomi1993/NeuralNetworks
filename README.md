# NeuralNetworks
This repositories documents different neural networks I have implemented during Machine Learning course I took as part of my Computer Science studies at Bar-Ilan University.


## Description

Neural Networks are Machine Learning computing models inspired by the biological neural networks that constitute animal brains.

A Neural Network (shortly NN) is based on a collection of connected computational units called neurons. Each connection can transmit a signal to other neurons. A neuron receives a signal then processes it and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer that is the input layer, through a number of hidden layers, to the last layer that is the output layer.

Here is an example of a simple NN with one hidden layer:  
<img src=https://user-images.githubusercontent.com/72878018/154792987-ae500000-2500-49ef-a7de-c8497ee20e5c.png width=30% height=30%>

And here is an example of a multi-hidden-layer NN, or a Deep Neural Network (DNN):  
<img src=https://user-images.githubusercontent.com/72878018/154793152-f4f6e957-304f-45ff-a11b-67eaafd57485.png width=60% height=60%>  
Note that the size of each layer (the number of nodes in each layer) can be different.

**Upnext**, I'll explain shortly about system requirements for running the programs, and then we'll discuss about three types of NNs:
1. ANN - Artificial Neural Network
2. DNN - Deep Neural Network
3. CNN - Convolutional Neural Network


## Requirements
To run the Neural Networks in this repository, please make sure your system meets the following requirements:
1. **Pyhon 3**. Tested on Python 3.8.
2. **Andconda 3**. A Python distribution that contains many usful libraries. Download: https://www.anaconda.com/products/individual
3. **PyTorch**. An open-source ML framework. Download: https://pytorch.org/get-started/locally/
4. **Python Packages**: numpy,  matplotlib, torch, torchvision, librosa and soundfile.


## ANN - Artificial Neural Network


### Description
In this part, you can find the implementation of my first simple Artificial Neural Network (ANN), implemented manually, without the use of any external packages other than NumPy. The ANN is trained on a dataset called “MNIST”, that contains grayscale images of 10 handwritten digits from 0 to 9. The task is to train a classifier that classifies this data. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel. This pixel-value is an
integer between 0 and 255. The labels are the numbers from 0 to 9.

![image](https://user-images.githubusercontent.com/72878018/154794399-53058676-b36a-4f21-b8a7-f0fe0ecff542.png)


### Instructions
1. Download the directory ANN from this repo: https://github.com/shlomi1993/NeuralNetworks/tree/main/ANN
2. Inside "ANN/data", extract the zip file "original.zip" to get the following files:
   -  train_x - Train objects. Each object is 28x28 values between 0 and 255.
   -  train_y - Train labels. Each label is an integer between 0 and 9. The i'th label associates with the i'th object.
   -  test_x - Test objects. Their labels needed to be predicted by the ANN.
3. To run the program, navigate to ANN directory and use the command:
   > python3 ann.py train_x train_y test_x
   
   where the argument x is the path to the x file.

**Important Note:** Running time on the whole dataset probably consumes a lot of time. For debugging purposes, use portion of the dataset. You can get a similar but smaller dataset by using the python script _data_sampler.py_ from this repo, or by using the reduced dataset from the directory "_ANN\data\reduced_".
  

### Full Report
For further reading about the NN's architecture, hyper-parameters, and loss per epochs, you can read the full report here:  
https://github.com/shlomi1993/NeuralNetworks/blob/main/ANN/ann_report.pdf
  

## DNN - Deep Neural Network

### Description

### PyTorch

### Instructions


## CNN - Convolutional Neural Network

### Description

### Convolution

### Instructions

## Notes

## Sources


