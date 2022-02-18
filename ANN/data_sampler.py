# Written by Shlomi Ben-Shushan


import sys
import numpy as np


def sample_data():
    """
    This function read the given data-files and creates much smaller data files
    out of them that can be used for debug.
    """

    # Read files.
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]

    # Create new debug train-data files.
    chosen = np.random.sample(range(55000), 500)
    chosen.sort()
    with open("train_x_debug", "w") as out1:
        with open(train_x, "r") as in1:
            for i, line in zip(range(500), in1):
                if i in chosen:
                    out1.write(line)
    with open("train_y_debug", "w") as out2:
        with open(train_y, "r") as in2:
            for i, line in zip(range(500), in2):
                if i in chosen:
                    out2.write(line)

    # Create new debug test-data file.
    chosen = np.random.sample(range(5500), 50)
    chosen.sort()
    with open("test_x_x_debug", "w") as out3:
        with open(test_x, "r") as in3:
            for i, line in zip(range(50), in3):
                if i in chosen:
                    out3.write(line)
