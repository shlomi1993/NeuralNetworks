# NeuralNetworks
This repository documents different neural networks I have implemented during Machine Learning course I took as part of my Computer Science studies at Bar-Ilan University.

![ActiveCourteousAmericanindianhorse-max-1mb](https://user-images.githubusercontent.com/72878018/154795472-e2d8cebc-d516-497c-9e7e-8fa66e2a41d1.gif)  
(GIF Source: https://www.3blue1brown.com/topics/neural-networks)


## Description

Neural Networks are Machine Learning computing models inspired by the biological neural networks that constitute animal brains.

A Neural Network (shortly NN) is based on a collection of connected computational units called neurons. Each connection can transmit a signal to other neurons. A neuron receives a signal then processes it and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer that is the input layer, through a number of hidden layers, to the last layer that is the output layer.

Here is an example of a simple NN with one hidden layer:  
<img src=https://user-images.githubusercontent.com/72878018/154792987-ae500000-2500-49ef-a7de-c8497ee20e5c.png width=30% height=30%>

And here is an example of a multi-hidden-layer NN, or a Deep Neural Network (DNN):  
<img src=https://user-images.githubusercontent.com/72878018/154793152-f4f6e957-304f-45ff-a11b-67eaafd57485.png width=60% height=60%>  
Note that the size of each layer (the number of nodes in each layer) can be different.

**Upnext**, I'll explain shortly about system requirements for running the programs, and then we'll discuss three types of NNs:
1. ANN - Artificial Neural Network
2. DNN - Deep Neural Network
3. CNN - Convolutional Neural Network


## Requirements

To run the Neural Networks in this repository, please make sure your system meets the following requirements:
1. **Pyhon 3.6+**. Tested on Python 3.8.
2. **Andconda 3**. A Python distribution that contains many usful libraries. Download: https://www.anaconda.com/products/individual
3. **PyTorch**. An open-source ML framework. Download: https://pytorch.org/get-started/locally/
4. **Python Packages**: numpy,  matplotlib, torch, torchvision, librosa and soundfile.


## ANN - Artificial Neural Network


### Description

In this part, you can find the implementation of my first simple Artificial Neural Network (ANN), implemented manually, without the use of any external packages other than NumPy.

#### Objects
The ANN is trained on a dataset called “MNIST”, which contains grayscale images of 10 handwritten digits from 0 to 9 and the task is to train a classifier that classifies this data. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel. This pixel-value is an integer between 0 and 255.

Here is an example of input objects:  
![image](https://user-images.githubusercontent.com/72878018/154794399-53058676-b36a-4f21-b8a7-f0fe0ecff542.png)


#### Labels
The labels are integers from 0 to 9.


### Instructions

1. Download the directory "ANN" from this repo: https://github.com/shlomi1993/NeuralNetworks/tree/main/ANN
2. Inside "ANN/data", extract the zip file "original.zip" to get the following files:
   -  train_x - Train objects. Each object is 28x28 values between 0 and 255.
   -  train_y - Train labels. Each label is an integer between 0 and 9. The i'th label associates with the i'th object.
   -  test_x - Test objects. Their labels needed to be predicted by the ANN.
3. To run the program, navigate to the ANN directory and use the command:
   > python3 ann.py train_x train_y test_x
   
   where the argument x is the path to the x file.

**Important Note:** Running time on the whole dataset probably consumes a lot of time. For debugging purposes, use a portion of the dataset. You can get a similar but smaller dataset by using the python script _data_sampler.py_ from this repo, or by using the reduced dataset from the directory "_ANN\data\reduced_".
  

### Full Report

For further reading about the NN's architecture, hyper-parameters, and loss per epoch, you can read the full report here:  
https://github.com/shlomi1993/NeuralNetworks/blob/main/ANN/ann_report.pdf
  

## DNN - Deep Neural Network


### Description

In this part, you can find my implementation of a multi-layer fully-connected neural network, or Deep Neural Network (DNN) using **PyTorch** package. To run this part, it is necessary to have PyTorch pre-installed on your machine.

#### Objects
The data in this part is called "FashionMNIST" and it contains 10 different categories of clothing. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel. This pixel-value is an integer between 0 and 255. 

Here is an example of input objects:  
![image](https://user-images.githubusercontent.com/72878018/154796283-43135c20-8841-4639-8165-6d73a6e1894f.png)

#### Labels
The possible labels are:  
0 - T-shirt/top  
1 - Trouser  
2 - Pullover  
3 - Dress  
4 - Coat  
5 - Sandal  
6 - Shirt  
7 - Sneaker  
8 - Bag  
9 - Ankle boo  

#### DNN Models
This part is divided into two sections:
1. **Experiments:** Implementation of six fully connected deep neural networks models for losses and accuracies tests.
2. **Best Model:** Implementation of a fully connected deep neural network model that obtains accuracy of at least 90%.


### Experiments

In this section, I have implemented six fully connected neural networks models via PyTorch. Each model has different settings and different effects in terms of loss and accuracy.

1. **Model A** - Neural Network with two hidden layers, the first layer is at size of 100 and the second layer is at size of 50. Both layers are followed by ReLU activation function, and the model is trained by **SGD** optimizer.

2. **Model B** - Neural Network with two hidden layers, the first layer is at size of 100 and the second layer is at size of 50. Both layers are followed by ReLU activation function, and this model is trained by **ADAM** optimizer.

3. **Model C - Dropout** – Similar to Model B, but with dropout layers on the output of each hidden layer.

4. **Model D - Batch Normalization** - Similar to Model B, but with Batch Normalization layer before each activation function invocation.

5. **Model E** - Neural Network with five hidden layers at sizes [128, 64, 10, 10, 10] using ReLU activation functions.

6. **Model F** - Neural Network with five hidden layers at sizes [128, 64, 10, 10, 10] using Sigmoid activation functions.

All models use _log_softmax_ as the output of the network and NLL loss function.

**Training:** The models are trained using FashionMNIST dataset for 10 epochs each. The calibration of the hyper-parameters is done by K-Fold Cross-Validation methodology.

**Note:** The main of the program does not run any of the above models/experiments, but you can change it as you wish to run them. I intention behind the models is to examine effects in terms of loss and accuracy per epoch and report them to the attached PDF report file. 


### Best Model

In this section, I have implemented a custom DNN named **Model S** to achieve an accuracy of at least 90%. It consists of two hidden layers, the first is half the size of the input, and the second is a quarter the size of the input. Each layer is followed by ReLU activation function and Batch Normalization, and Drop-out is activated after it. In the end, the DNN returns output through log_softmax function. This model is trained by Adam optimizer as long as the validation accuracy does not converge, and by SGD as long as it converges.

To run the model, follow the instructions below.


### Instructions

1. Download the directory "DNN" from this repo: https://github.com/shlomi1993/NeuralNetworks/tree/main/DNN
2. Inside "DNN/data", extract the zip file "original.zip" to get the following files:
   -  train_x - Train objects. Each object is 28x28 values between 0 and 255.
   -  train_y - Train labels. Each label is an integer between 0 and 9 that represents a clothing item. The i'th label associated with the i'th object.
   -  test_x - Test objects. Their labels needed to be predicted by the DNN.
3. To run the program, navigate to the DNN directory and use the command:
   > python3 dnn.py train_x train_y test_x test_y
   
   where the argument x is the path to the x file. The last argument is a path to a new **output log file**.

**Important Note:** Running time on the whole dataset probably consumes a lot of time. For debugging purposes, use a portion of the dataset. You can get a similar but smaller dataset by using the python script _data_sampler.py_ from this repo, or by using the reduced dataset from the directory "_DNN\data\debug_".
  

### Full Report

For further reading about the NN's architecture, hyper-parameters, and loss/accuracy per epoch, you can read the full report here:  
https://github.com/shlomi1993/NeuralNetworks/blob/main/DNN/dnn_report.pdf


## CNN - Convolutional Neural Network

### Description

In this part, I have implemented a Convolutional Neural Network (CNN)., or Deep Neural Network (DNN) using **PyTorch** package.

![image](https://user-images.githubusercontent.com/72878018/154833685-918b2e21-253d-4fdb-b44c-6f3d3f158678.png)

#### What is Convolution?
> In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of artificial neural network, most commonly applied to analyze visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation equivariant responses known as feature maps. Counter-intuitively, most convolutional neural networks are only equivariant, as opposed to invariant, to translation. They have applications in image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing,[6] brain-computer interfaces, and financial time series.  
(Source: https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### Objects
The CNN model is trained to classify a speech command using speech data. The dataset contains 30 different categories of commands, and the task is to train a classifier that classifies this data. Each speech utterance is ∼1 sec long that can be read by PyTorch dataset reader called _gcommand_loader.py_. 

#### Lables:
Each object can be assign to one of the following labels:  
bed, cat, down, five, go, house, marvin, no, on, right, sheila, stop, tree, up, yes, bird, dog, eight, four, happy, left, nine, off, one, seven, six, three, two, wow, zero


### Instructions

1. Download the directory "CNN" from this repo: https://github.com/shlomi1993/NeuralNetworks/tree/main/CNN
2. Download the dataset from:https://drive.google.com/file/d/1vAyAPNeSwwx499eC9Jb1zAfuqJHs-S4Q/view
   - Note that the dataset is very heavy. For debugging, you can use the reduced dataset from the dictionary: "_CNN/data/data_debug_".
   - The training data is already separated into training and validation.
   - The **training dataset** is in form of a file-system directory that contains different subdirectories named after the labels in the experiment, and each sub-directory contains samples as wav files, corresponding to the name of the directory (which is the label).
   - The **validation dataset** is in the same form as the form of the training dataset.
   - The **test dataset** is in the form of a file-system directory that contains another directory (whose name doesn't matter), that contains test samples as wav files in names that consist of integers only.
3. To run the program, navigate to the CNN directory and use the command:
   > python3 cnn.py data/train data/valid data/test test_y
4. Alternatively, to make the program create plot files, use the command:
   > python ex5.py data/train data/valid data/test test_y --plot
   
   where the argument x is the path to the x file. The last argument is a path to a new **output log file**.

**Important Note:** Running time on the whole dataset probably consumes a lot of time. For debugging purposes, use a portion of the dataset. You can get a similar but smaller dataset by using the reduced dataset from the directory "_CNN\data\data_debug_".
  

### Full Report

For further reading about the NN's architecture, hyper-parameters, and loss/accuracy per epoch, you can read the full report here:  
https://github.com/shlomi1993/NeuralNetworks/blob/main/CNN/report/cnn_report.pdf


## Sources
1. Lecture Notes from the Machine Learning lectures of Professor Yossi Keshet at Bar-Ilan University.
2. Wikipedia: https://www.wikipedia.org/
3. 3Blue1Brown: https://www.3blue1brown.com/topics/neural-networks
