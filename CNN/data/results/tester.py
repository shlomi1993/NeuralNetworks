import sys


my_results_file = sys.argv[1]
file = open(my_results_file, 'r')
test_y = file.read().split('\n')
file.close()
file = open('test_y_correct_labels', 'r')
test_labels = file.read().split('\n')
file.close()
hits = 0
N = len(test_y)
if N != len(test_labels):
    print("Wrong number of predictions!")
else:
    for y, y_hat in zip(test_y, test_labels):
        if y == y_hat:
            hits += 1
print(f'{my_results_file} accuracy is {str(round(hits * 100. / N, 2))}%')
