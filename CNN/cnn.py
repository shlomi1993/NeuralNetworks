# Shlomi Ben-Shushan 311408264


import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gcommand_loader import GCommandLoader


class SoundClassifierNN(nn.Module):
    """
    This class is an implementation of a multi-class neural network for sound
    classification. It consists of five convolution layers at sizes 16, 32, 64,
    128 and 256, and two linear (fully-connected) layers at sizes 512 and 128.
    Each convolution layer activates Batch Normalization, ReLU and Max Pooling
    after the convolution, and each linear layer activates Batch Normalization,
    ReLU and Dropout after it.
    """
    def __init__(self, classes):
        """
        NN class constractor.
        :param classes: a list of the classes in the experiment.
        """
        super(SoundClassifierNN, self).__init__()
        self.classes = classes
        def create_convolution_layer(in_channels, out_channels, kernel_size,
                                     stride, pad, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel),
            )
        def create_linear_layer(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(),
            )
        self.conv1 = create_convolution_layer(1, 16, 5, 2, 2, (2, 2))
        self.conv2 = create_convolution_layer(16, 32, 3, 1, 1, (2, 2))
        self.conv3 = create_convolution_layer(32, 64, 3, 1, 1, (2, 2))
        self.conv4 = create_convolution_layer(64, 128, 3, 1, 1, (2, 2))
        self.conv5 = create_convolution_layer(128, 256, 3, 1, 1, (2, 2))
        self.linear1 = create_linear_layer(512, 128)
        self.linear2 = create_linear_layer(128, len(classes))
        print('Sound Classifier Neural Network model created.')

    def forward(self, x):
        """
        Network's forward method, implements learning through the layers.
        :param x: object to classify.
        :return: y_hat prediction.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = F.log_softmax(out, -1)
        return out

    @staticmethod
    def __plot(t_losses, v_losses, t_accuracies, v_accuracies):
        """
        This method creates a losses and accuracies graphs.
        It is a private method because it was designed uniquely for this model.
        :param t_losses: a list of training losses.
        :param v_losses: a list of validation losses.
        :param t_accuracies: a list of training accuracies.
        :param v_accuracies: a list of validation accuracies.
        :return: None, but it save graphs to PNG files.
        """
        plt.figure()
        plt.title('SCNN - Average loss per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss values')
        plt.plot(list(range(len(t_losses))), t_losses, v_losses)
        plt.legend(["Training Loss", "Validation Loss"])
        plt.savefig("SCNN_Loss_per_Epoch.png")
        plt.figure()
        plt.title('SCNN - Average accuracy per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy percentages')
        plt.plot(list(range(len(t_accuracies))), t_accuracies, v_accuracies)
        plt.legend(["Training Accuracy", "Validation Accuracy"])
        plt.savefig("SCNN_Accuracy_per_Epoch.png")

    def __train_epoch(self, optimizer, train_dataset):
        """
        This method trains the model for one epoch within the training phase.
        This method is private because it is for internal use only.
        :param optimizer: to update model's parameters according to the loss
        :param train_dataset: the training dataset in a DataLoader form.
        :return: training loss and accuracy as floats.
        """
        self.train()
        train_loss = 0
        correct = 0
        for x, y in train_dataset:
            optimizer.zero_grad()
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = y_hat.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).cpu().sum().item()
        train_loss /= (len(train_dataset))
        train_accuracy = (100. * correct) / len(train_dataset.dataset)
        return train_loss, train_accuracy

    def __validate_epoch(self, valid_dataset):
        """
        This method validates the model for one epoch within the training phase.
        This method is private because it is for internal use only.
        :param valid_dataset: the validation dataset in a DataLoader form.
        :return: validation loss and accuracy as floats.
        """
        self.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in valid_dataset:
                y_hat = self(x)
                valid_loss += F.nll_loss(y_hat, y, reduction="sum").item()
                pred = y_hat.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).cpu().sum().item()
        valid_loss /= len(valid_dataset.dataset)
        valid_accuracy = (100. * correct) / len(valid_dataset.dataset)
        return valid_loss, valid_accuracy

    def learn(self, train_gc, valid_gc, optimizer, n_epochs, save_plot=False):
        """
        This method implements the NN learning (or training) phase by running
        training and validation for number of epochs.
        :param train_gc: training dataset in a GCommands form.
        :param valid_gc: validation dataset in a GCommands form.
        :param optimizer: to update model's parameters according to the loss
        :param n_epochs: the number of epochs to train.
        :param save_plot: a boolean that tells if to create plot PNGs or not.
        :return: None, but it changes model's parameters and can create graphs.
        """
        train_dataset = DataLoader(train_gc, batch_size=64, shuffle=True)
        valid_dataset = DataLoader(valid_gc, batch_size=64, shuffle=True)
        t_losses, t_accuracies = [], []
        v_losses, v_accuracies = [], []
        print('Training epochs...')
        for e in range(n_epochs):
            print(f'epoch {e + 1}/{n_epochs}:', end=' ')

            # Train:
            t_loss, t_accuracy = self.__train_epoch(optimizer, train_dataset)
            t_losses.append(t_loss)
            t_accuracies.append(t_accuracy)
            print(f'train loss: {str(round(t_loss, 4))}', end=', ')
            print(f'train accuracy: {str(round(t_accuracy, 2))}%', end=', ')

            # Validate:
            v_loss, v_accuracy = self.__validate_epoch(valid_dataset)
            v_losses.append(v_loss)
            v_accuracies.append(v_accuracy)
            print(f'valid loss: {str(round(v_loss, 4))}', end=', ')
            print(f'valid accuracy: {str(round(v_accuracy, 2))}%')

        print('Training completed!')
        if save_plot:
            self.__plot(t_losses, v_losses, t_accuracies, v_accuracies)
            print('Charts of losses and accuracies were saved to PNG files.')

    def predict(self, test_gc, log_file):
        """
        This function implements the NN test phase by inserting new unseen
        objects to the trained model, interpreting the results, and write them
        to an output log file.
        :param test_gc: test objects in a GCommands form.
        :param log_file: output log file path/name.
        :return: a list of samples-predictions tuples, and creates a log file.
        """
        # This try-except is a solution for a problem I've encountered where in
        # PyCharm IDE the tuples in dataset.spects where in the shape of
        # (.../...\\test\\filename.wav, 0) while in Google Colab and WSL they
        # were in the shape of (.../.../test/filename.wav, 0).
        try:
            samples = [tup[0].split('\\')[2] for tup in test_gc.spects]
        except IndexError:
            samples = [tup[0].split('/')[3] for tup in test_gc.spects]
        test_dataset = DataLoader(test_gc)
        print('Calculating predictions...')
        predictions = []
        for samp, (x, y) in zip(samples, test_dataset):
            y_hat = self(x).max(1, keepdim=True)[1].item()
            pred = self.classes[y_hat]
            predictions.append((samp, pred))
        predictions = sorted(predictions, key=lambda t: int(t[0].split('.')[0]))
        with open(log_file, 'w') as out:
            for (samp, pred) in predictions:
                out.write(f'{samp},{pred}\n')
        print(f'Predictions were successfully logged to the file "{log_file}".')


def main():
    """
    This is the entry point of the program.
    :return: None.
    """

    # Expecting paths for train, valid and test directories, and log file name.
    n_args = len(sys.argv)
    if n_args < 5:
        raise 'Not enough inputs.'

    # If a plot flag was given, the program will save plots as PNG files.
    p_flag = (n_args == 6) and (sys.argv[5] == '-p' or sys.argv[5] == '--plot')

    # Load GCommands.
    train_gc = GCommandLoader(sys.argv[1])
    valid_gc = GCommandLoader(sys.argv[2])
    test_gc = GCommandLoader(sys.argv[3])

    # Get a list of classes from the training GCommands dataset.
    # The try-except is used because of the same reason as described above.
    try:
        classes = [tup[0].split('\\')[1] for tup in train_gc.spects]
    except IndexError:
        classes = [tup[0].split('/')[2] for tup in train_gc.spects]
    classes = sorted(list(set(classes)))

    # Create a SCNN model and Adam optimizer.
    model = SoundClassifierNN(classes)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Make the model learn and predict.
    model.learn(train_gc, valid_gc, optim, n_epochs=20, save_plot=p_flag)
    model.predict(test_gc, log_file=sys.argv[4])
    print('Done!')


main()
