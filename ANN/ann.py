# Written by Shlomi Ben-Shushan


import sys
import numpy as np
import matplotlib.pylab as plt


# Globals: There are 784 (= 28 * 28) inputs and 10 outputs (0-9).
INPUTS = 784
OUTPUTS = 10
EPSILON = 1e-6


def preprocess(train_x, train_y, test_x):
    """
    This function normalizes the given objects and encode the given targets
    using One-Hot encoding.

    :param train_x: training objects.
    :param train_y: training targets correspondent to the training objects.
    :param test_x: test objects.
    :return normalized objects and one-hot encoded targets.
    """

    # Normalize objects:
    normalized_train_x = train_x / 255.
    normalized_test_x = test_x / 255.

    # One-Hot encode targets:
    encoded_train_y = []
    for y in train_y:
        encoded_y = np.zeros(OUTPUTS)
        encoded_y[y] = 1.
        encoded_train_y.append(encoded_y)
    encoded_train_y = np.array(encoded_train_y)

    # Return preprocessed data.
    return normalized_train_x, encoded_train_y, normalized_test_x


def shuffle(objects, targets):
    """
    This function gets a data-set consists of objects and correspondent targets
    and returns a shuffled data-set of objects and correspondent targets.

    :param objects: x vectors of the data-set.
    :param targets: y values where the i"th y correspondent to the i'th x.
    :return shuffled data-set (objects and targets).
    """
    
    data = list(zip(objects, targets))
    np.random.shuffle(data)
    shuffled_x, shuffled_y = zip(*data)
    return np.array(shuffled_x), np.array(shuffled_y)


def init_params(layer_size):
    """
    This function initializes parameters for the NN w.r.t the number of inputs,
    number of outputs (that are constant in this program) and the number of
    nodes in the hidden layer (which is a hyper-parameter in this program).
    Note that the function normalizes the models by dividing by the square root
    of the number of nodes in their level.

    :param layer_size: the number of nodes in the hidden layer.
    :return a dictionary contains two models and two biases.
    """
    
    W1 = np.random.randn(layer_size, INPUTS) / np.sqrt(INPUTS)
    b1 = np.zeros((layer_size, 1))
    W2 = np.random.randn(OUTPUTS, layer_size) / np.sqrt(layer_size)
    b2 = np.zeros((OUTPUTS, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def sigmoid(x):
    """
    An implementation of sigmoid activation function.

    :param x: a float or a vector of floats.
    :return sigmoid result.
    """
    
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    An implementation of softmax function activation.

    :param x: a float or a vector of floats.
    :return softmax result.
    """
    
    x -= np.max(x)
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=0, keepdims=True)


def predict(x, params):
    """
    This function gets objects and predicts their targets using forward propagation.

    :param x: new unseen objects.
    :param params: dictionary of parameters for fprop function.
    :return a vector of predictions (test_y).
    """
    
    cache = fprop(x, params)
    return np.argmax(cache["h2"], axis=0)


def fprop(x, params):
    """
    An implementation of forward-propagation process.

    :param x: vector of objects (train or test).
    :param params: dictionary of parameters required for the process.
    :return a vector of predictions (test_y).
    """
    
    W1, b1, W2, b2 = [params[key] for key in params.keys()]
    z1 = np.dot(W1, x.T) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2}
    for key in params:
        cache[key] = params[key]
    return cache


def bprop(y, cache):
    """
    An implementation of back-propagation process.

    :param y: vector of test-targets.
    :param cache: dictionary contains all the data needed for the calculations.
    :return a dictionary with the derivatives of the parameters.
    """
    
    x, y_hat, h1, W2, z1 = [cache[key] for key in ("x", "h2", "h1", "W2", "z1")]
    n = x.shape[0]
    dz2 = y_hat - y.T
    dW2 = np.dot(dz2, h1.T) / n
    db2 = np.sum(dz2, axis=1, keepdims=True) / n
    dz1 = np.dot(W2.T, dz2) * h1 * (1 - h1)
    dW1 = 1. / n * dz1.dot(x)
    db1 = 1. / n * np.sum(dz1, axis=1, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(train_x, train_y, epochs, eta, layer_size):
    """
    An implementation of the Training process of the NN.

    :param train_x: a set of training objects.
    :param train_y: a set of correspondent training targets.
    :param epochs: a hyper-parameter -- the number of iterations.
    :param eta: a hyper-parameter -- the learning rate.
    :param layer_size: a hyper-parameter -- number of nodes in the hidden layer.
    :return a dictionary with the derivatives of the parameters, and losses.
    """
    
    np.random.seed(1)
    params = init_params(layer_size)
    losses = []
    for t in range(epochs):

        # 1) Shuffle:
        objects, targets = shuffle(train_x, train_y)

        # 2) Forward-Propagation:
        cache = fprop(objects, params)

        # 3) Loss Calculation:
        loss = -np.mean(targets * np.log(cache["h2"].T + EPSILON))
        losses.append(loss)

        # 4) Back-Propagation:
        derivatives = bprop(targets, cache)

        # 5) Update parameters using Gradient-Decent.
        params["W1"] = params["W1"] - eta * derivatives["W1"]
        params["b1"] = params["b1"] - eta * derivatives["b1"]
        params["W2"] = params["W2"] - eta * derivatives["W2"]
        params["b2"] = params["b2"] - eta * derivatives["b2"]

    return params, losses


def make_plot(losses):
    """
    Creates a graph that describes the loss reduction along the epochs.

    :param losses: array contains the loss in each iteration of the NN.
    :return None, but it shows a graph using matplotlib.
    """
    
    plt.figure()
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def create_validation_sets(objects, targets):
    """
    This function creates 5 validation sets according to K-Fold
    cross-validation technique with k=5.

    :param objects: a set of objects, the x vectors of the data-set.
    :param targets: a set of targets, the y values of the data-set.
    :return 5 new data-sets and correspondent test-sets.
    """

    # Divide data to 5 chunks.
    train_x, train_y, test_x, test_y = [], [], [], []
    size = int(0.2 * len(objects))
    o_chunks = [objects[i:i + size] for i in range(len(objects))[::size]]
    t_chunks = [targets[i:i + size] for i in range(len(targets))[::size]]

    # Each time determine 4 chunks as new train-set the fifth as a test-set.
    for i in range(5):
        a = i
        b = (i + 1) % 5
        c = (i + 2) % 5
        d = (i + 3) % 5
        e = (i + 4) % 5
        train_x.append(np.concatenate([o_chunks[a], o_chunks[b], o_chunks[c],
                                       o_chunks[d]], axis=0))
        train_y.append(np.concatenate([t_chunks[a], t_chunks[b], t_chunks[c],
                                       t_chunks[d]], axis=0))
        test_x.append(o_chunks[e])
        test_y.append(t_chunks[e])

    # Return 5 data-sets and 5 test-sets (each array contains 5 sub-arrays).
    return train_x, train_y, test_x, test_y


def predict_accuracy(x, y, params):
    """
    This function gets objects and predicts their targets using forward prop.
    Then, it compares the predicted targets to the targets given in y input and
    return the accuracy as the number of hits (y_hat == y) divide by total.

    :param x: seen test objects.
    :param y: seen test targets correspondent to the test objects.
    :param params: dictionary of parameters for fprop function.
    :return a float that represents accuracy.
    """
    
    cache = fprop(x, params)
    y_hat = np.argmax(cache["h2"], axis=0)
    y = np.argmax(y, axis=1)
    return (y_hat == y).mean() * 100


def cross_validate(validation_sets, epochs, eta, layer_s, print_info):
    """
    This function runs the NN on the validation-sets and returns avg accuracy.

    :param validation_sets: 5 train-sets and 5 test sets.
    :param epochs: a hyper-parameter -- the number of iterations.
    :param eta: a hyper-parameter -- the learning rate.
    :param layer_s: a hyper-parameter -- number of nodes in the hidden layer.
    :param print_info: a boolean that tells if to print information or not.
    :return prints essential data and returns the average of 5 accuracies.
    """
    
    train_x, train_y, test_x, test_y = validation_sets
    accuracies = []
    for i in range(5):
        parameters, losses = train(train_x[i], train_y[i], epochs, eta, layer_s)
        acc = predict_accuracy(test_x[i], test_y[i], parameters)
        accuracies.append(acc)
    average = round(np.average(accuracies), 4)
    if print_info:
        print(f"{epochs}/{eta}/{layer_s}: Current accuracy: {average}")
    return average


def calibrate(epochs, etas, layer_sizes, print_bests, print_all):
    """
    This function finds the hyper-parameters from the given range that gives the
    highest accuracy when using them in the NN.

    :param epochs: range of potential epochs.
    :param etas: range of potential etas.
    :param layer_sizes: range of potential layer_sizes.
    :param print_bests: a boolean that tells to print the current best HP.
    :param print_all: a boolean that tells to print info in every iteration.
    :return prints essential info and returns the best HP and accuracy.
    """
    print("Running calibration...")

    # Get (debug) data:
    train_x = np.loadtxt("train_x_debug")
    train_y = np.loadtxt("train_y_debug", dtype=int)
    test_x = np.loadtxt("test_x_debug")

    # Preprocessing:
    train_x, train_y, test_x = preprocess(train_x, train_y, test_x)
    objects, targets = shuffle(train_x, train_y)
    v_sets = create_validation_sets(objects, targets)

    # Traverse hyper-parameters ranges to find the best ones.
    best_epo, best_eta, best_lay, best_acc = 0, 0, 0, -1
    for epo in epochs:
        for eta in etas:
            for layer_size in layer_sizes:
                acc = cross_validate(v_sets, epo, eta, layer_size, print_all)
                if best_acc < acc:
                    best_epo, best_eta = epo, eta
                    best_lay, best_acc = layer_size, acc
                    if print_bests:
                        print(f"Best so far: epochs={epo}, eta={eta},"
                              f"layer_size={layer_size} => accuracy:", acc)
    print("Done.")
    return best_epo, best_eta, best_lay, best_acc


def main():
    """
    This function is the entry point of the program.
    """

    # Get data:
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2], dtype=int)
    test_x = np.loadtxt(sys.argv[3])

    # Preprocessing:
    train_x, train_y, test_x = preprocess(train_x, train_y, test_x)

    # Train and predict using the best parameter found earlier:
    params, losses = train(train_x, train_y, epochs=500, eta=0.4, layer_size=100)
    predictions = predict(test_x, params)

    # Create loss graph:
    make_plot(losses)

    # Output to a file:
    with open("test_y", "w+") as log:
        for y in predictions:
            log.write(str(y) + "\n")


# Here is an example of a way to find best hyper-parameters using calibration:
# epochs_rng = range(100, 1050, 50)
# etas_rng = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# layer_sizes_rng = range(50, 160, 10)
# ep, et, ly, ac = calibrate(epochs_rng, etas_rng, layer_sizes_rng, print_bests=True, print_all=True)

main()
