import logRegres
import numpy as np


def classify_vector(inX, weights):
    prob = logRegres.sigmoid(sum(inX*weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    train_dataset = open(r"horseColicTraining.txt")
    test_dataset = open(r"horseColicTest.txt")
    training_set, training_label = [], []
    for line in train_dataset.readlines():
        curr_line = line.strip().split("\t")
        line_arr = []
        for i in range(len(curr_line)-1):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_label.append(float(curr_line[-1]))
    train_weights, weights_array = logRegres.stoc_grad_ascent1(np.array(training_set), training_label, 500)
    error_count = 0
    num_test_vec = 0.0
    for line in test_dataset.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split("\t")
        line_arr = []
        for i in range(len(curr_line)-1):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[-1]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vec)
    print("the error rate of this test is: {}".format(error_rate))
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after {} iterations the average error rate is:{}".format(num_tests, error_sum/float(num_tests)))


if __name__ == "__main__":
    multi_test()

