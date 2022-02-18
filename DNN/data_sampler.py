# Written by Shlomi Ben-Shushan


import random


def sample_data():
    """
    This function read the given data-files and creates much smaller data files
    out of them that can be used for debug and development.
    """

    # Create debug train-data files:
    chosen = random.sample(range(55000), 5000)
    chosen.sort()
    with open("train_x_debug", "w") as out1:
        with open("train_x", "r") as in1:
            for i, line in zip(range(55000), in1):
                if i in chosen:
                    out1.write(line)
    with open("train_y_debug", "w") as out2:
        with open("train_y", "r") as in2:
            for i, line in zip(range(55000), in2):
                if i in chosen:
                    out2.write(line)

    # Create debug test-data file:
    chosen = random.sample(range(5000), 100)
    chosen.sort()
    with open("test_x_debug", "w") as out3:
        with open("test_x", "r") as in3:
            for i, line in zip(range(5000), in3):
                if i in chosen:
                    out3.write(line)


sample_data()
