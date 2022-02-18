# Written by Shlomi Ben-Shushan


import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])


class ModelA(nn.Module):
    """
    Model A - NN with two hidden layers where the first layer is in size of 100
    and the second layer is in size of 50, and both layers followed by ReLU
    activation function. This model is trained with SGD optimizer.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, n_outputs)
        self.optimizer = optim.SGD(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), dim=1)


class ModelB(nn.Module):
    """
    Model B - NN with two hidden layers where the first layer is in size of 100
    and the second layer is in size of 50, and both layers followed by ReLU
    activation function. This model is trained with ADAM optimizer.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), dim=1)


class ModelC(nn.Module):
    """
    Model C - Similar to Model B, but with drop-out.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.fc0(out))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        return F.log_softmax(self.fc2(out), dim=1)


class ModelD(nn.Module):
    """
    Model D - Similar to Model B. but with Batch Normalization.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, n_outputs)
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)
        self.optimizer = optim.Adam(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.bn0(self.fc0(out)))
        out = F.relu(self.bn1(self.fc1(out)))
        # out = self.bn0(F.relu(self.fc0(out)))
        # out = self.bn1(F.relu(self.fc1(out)))
        return F.log_softmax(self.fc2(out), dim=1)


class ModelE(nn.Module):
    """
    Model E - NN with 5 hidden-layers using ReLU and ADAM optimizer.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.fc0(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        return F.log_softmax(self.fc5(out), dim=1)


class ModelF(nn.Module):
    """
    Model F - NN with 5 hidden-layers using Sigmoid and SGD optimizer.
    """

    def __init__(self, image_size, n_outputs, eta):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = torch.sigmoid(self.fc0(out))
        out = torch.sigmoid(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        out = torch.sigmoid(self.fc4(out))
        return F.log_softmax(self.fc5(out), dim=1)


class ModelS(nn.Module):
    """
    Model S - This is a custom NN I have developed to achieve accuracy over 90%.
    it consists of two hidden layers, the first is half the size of the input,
    and the second is quarter the size of the input. Each layer is followed by
    ReLU activation function and Batch Normalization, and drop-out is activated
    after it. In the end, the NN returns output through log_softmax function.
    This model is trained by Adam optimizer as long as the validation accuracy
    does not converge, and by SGD as long as it converges.
    """

    def __init__(self, image_size, n_outputs, adam_eta, sgd_eta):
        super(ModelS, self).__init__()
        self.image_size = image_size
        layer1_size = int(0.5 * image_size)
        layer2_size = int(0.5 * layer1_size)
        self.fc0 = nn.Linear(image_size, layer1_size)
        self.fc1 = nn.Linear(layer1_size, layer2_size)
        self.fc2 = nn.Linear(layer2_size, n_outputs)
        self.bn0 = nn.BatchNorm1d(layer1_size)
        self.bn1 = nn.BatchNorm1d(layer2_size)
        self.optimizer = optim.Adam(self.parameters(), lr=adam_eta)
        self.optimizer2 = optim.SGD(self.parameters(), lr=sgd_eta)

    def forward(self, x):
        out = x.view(-1, self.image_size)
        out = F.relu(self.bn0(self.fc0(out)))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.dropout(out, training=self.training)
        return F.log_softmax(self.fc2(out), dim=1)


def create_loaders(train_x_file, train_y_file):
    """
    This function reads data from the given files and create Tensor DataLoaders
    out of them for training and validation phases of a NN.

    :param train_x_file: path to train_x file.
    :param train_y_file: path to train_y file.
    :return: Two DataLoaders with batch_size=64 for training and validation.
    """

    # Load text from files.
    train_x_np = np.loadtxt(train_x_file)
    train_y_np = np.loadtxt(train_y_file)

    # Split indexes randomly to groups at sizes 80% and 20% of the training-set.
    n_train = len(train_x_np)
    indexes = [int(i) for i in list(range(n_train))]
    np.random.shuffle(indexes)
    s = int(0.2 * n_train)
    train_idx = indexes[s:]
    valid_idx = indexes[:s]

    # Create new training and validation numpy arrays.
    train_x = np.array([train_x_np[i] for i in train_idx])
    train_y = np.array([train_y_np[i] for i in train_idx])
    valid_x = np.array([train_x_np[i] for i in valid_idx])
    valid_y = np.array([train_y_np[i] for i in valid_idx])

    # Convert numpy arrays to normalized tensors.
    train_x_t = transform(train_x / 255.)[0].float()
    train_y_t = torch.from_numpy(train_y).long()
    valid_x_t = transform(valid_x / 255.)[0].float()
    valid_y_t = torch.from_numpy(valid_y).long()

    # Create tensor datasets.
    train_dataset = TensorDataset(train_x_t, train_y_t)
    valid_dataset = TensorDataset(valid_x_t, valid_y_t)

    # Create and return data loaders (with batch_size=64).
    return DataLoader(train_dataset, 64), DataLoader(valid_dataset, 64)


def convergence_check(accuracies):
    """
    This function tells if the accuracies are converges by checking the recent
    values. If the difference between the maximal value of among them to the
    minimal value among them is greater than 1, then the accuracies are not
    converges. Otherwise, they are converges.

    :param accuracies: a list of accuracies in floats of tensors.
    :return: True if the recent 5 values are similar, False Otherwise.
    """
    
    if len(accuracies) < 20:
        return False
    recent = accuracies[-5:]
    return (max(recent).item() - min(recent).item()) <= 0.5


def train_epoch(model, train_loader, converged):
    """
    This function trains one epoch within the CNN training phase.

    :param converged:
    :param model: the model that needs to be trained.
    :param train_loader: training dataset as DataLoader object.
    :return: loss and accuracy floats, and a trained model.
    """
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        model.optimizer.zero_grad()
        if hasattr(model, 'optimizer2'):
            model.optimizer2.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if hasattr(model, 'optimizer2') and converged:
            model.optimizer2.step()
        else:
            model.optimizer.step()
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    train_loss /= (len(train_loader))
    train_acc = ((100. * correct) / (len(train_loader.dataset)))
    return train_loss, train_acc


def validate_epoch(model, valid_loader):
    """
    This function validates one epoch within the CNN training phase.

    :param model: the model that needs to be validated.
    :param valid_loader: validation dataset as DataLoader object.
    :return: loss and accuracy floats, and a validated model.
    """
    
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    valid_loss /= len(valid_loader.dataset)
    valid_acc = (100. * correct) / (len(valid_loader.dataset))
    return valid_loss, valid_acc


def plot(title, values1, values2, descriptions):
    """
    This function creates a graph from the given data and show it.

    :param title: title of the graph.
    :param values1: values to the first plot-line.
    :param values2: values to the second plot-line
    :param descriptions: descriptions fo the plot-legend.
    """
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.plot(np.arange(len(values1)), values1, values2)
    plt.legend(descriptions)
    plt.show()


def train_model(model, train_x_file, train_y_file, n_epochs=10, create_plot=False):
    """
    This function is the implementation of the training phase of the CNN.

    :param model: the model that needs to be trained.
    :param train_x_file: path to train_x file.
    :param train_y_file: path to train_y file.
    :param n_epochs: number of epochs for training. default is 10.
    :param create_plot: a boolean that tells whether to create a plot or not.
    :return: average accuracies, and makes the model ready for testing.
    """

    # Get loaders and initialize arrays to store losses and accuracies.
    train_loader, valid_loader = create_loaders(train_x_file, train_y_file)
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    # In each epoch, train-validate the model and save losses and accuracies.
    converged = False
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, converged)
        valid_loss, valid_accuracy = validate_epoch(model, valid_loader)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        converged = convergence_check(valid_accuracies)

    # Optional: create plots and show graphs.
    if create_plot:
        name = str(model).split("(")[0]
        title = f"{name} - Average loss per epoch"
        legend = ["Training Loss", "Validation Loss"]
        plot(title, train_losses, valid_losses, legend)
        title = f"{name} - Average accuracy per epoch"
        legend = ["Training Accuracy", "Validation Accuracy"]
        plot(title, train_accuracies, valid_accuracies, legend)

    # Return average training accuracy and average validation accuracy,
    return np.average(train_accuracies), np.average(valid_accuracies)


def test_model(model, test_x_file, output_log_name):
    """
    This function is the implementation of the testing phase of the CNN.

    :param model: a model ready to test new unseen objects.
    :param test_x_file: path to test_x file (unseen objects).
    :param output_log_name: path to output log file.
    """
    
    test_x = transform(np.loadtxt(test_x_file) / 255.)[0].float()
    with open(output_log_name, "w+") as out:
        for x in test_x:
            out.write(str(model(x).max(1, keepdim=True)[1].item()) + "\n")


def find_optimal_learning_rate(model_init):
    """
    This is a debug functions that helps to focus on one model and find the
    learning-rate hyper-parameter (eta) that gives the most accurate results.

    :param model_init: a model constructor function.
    :return: the eta that gives the highest accuracy.
    """
    name = model_init.__name__
    print(f"Testing {name}...")
    eta = 0.001
    best = (-1., 0.)
    end = 1.0
    while eta < end:
        model = model_init(784, 10, eta)
        _, valid_acc = train_model(model, "train_x_debug", "train_y_debug")
        if best[1] < valid_acc:
            best = (eta, valid_acc)
        print(f"{name} test progress: {int(eta / end)}%.\t\t"
              f"Current: eta={eta}, accuracy={str(round(valid_acc, 4))}%\t\t"
              f"Best: eta={best[0]}, accuracy={str(round(best[1], 4))}%")
        eta = round(eta + 0.001, 4)
    print("Done!")
    print(f"Best eta for {name} is {best[0]} with accuracy of {best[1]}%\n")
    return best[0]


# This is how I found the optimized learning-rate for each model:
# eta_a = find_optimal_learning_rate(ModelA)  # Found best eta_a = 0.216.
# eta_b = find_optimal_learning_rate(ModelB)  # Found best eta_b = 0.005.
# eta_c = find_optimal_learning_rate(ModelC)  # Found best eta_c = 0.005.
# eta_d = find_optimal_learning_rate(ModelD)  # Found best eta_d = 0.005.
# eta_e = find_optimal_learning_rate(ModelE)  # Found best eta_e = 0.013.
# eta_f = find_optimal_learning_rate(ModelF)  # Found best eta_f = 0.023.


def main():
    model = ModelS(image_size=784, n_outputs=10, adam_eta=0.005, sgd_eta=0.2)
    if len(sys.argv) == 1:
        t, v = train_model(model, "train_x_debug", "train_y_debug", 100, True)
        t, v = str(round(t, 4)), str(round(v, 4))
        print(f"average training accuracy: {t}, average validation accuracy: {v}")
    elif len(sys.argv) > 4:
        train_model(model, sys.argv[1], sys.argv[2], n_epochs=100)
        test_model(model, sys.argv[3], sys.argv[4])
    else:
        print("Error: Not enough arguments.")


main()
