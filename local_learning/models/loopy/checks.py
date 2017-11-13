# all checks return a float from 0 to 1, saying how well this check did
# we maybe keep track of all checks performed, and then can query by different checks
import traceback
import local_learning
import logging
logger = logging.getLogger()

def check_signal_propagation(harness_class):
    # check that we can get a forward and backward pass within a reasonable amount of time on a randomly intialized model
    harness = harness_class(input_size=2, output_size=1)

    _, forward_info = harness.forward([5], max_steps=75)
    _, backward_info = harness.backward([5], max_steps=75)

    if forward_info['steps'] + backward_info['steps'] > 140:
        return 0.0

    return 0.5 + (forward_info['steps'] + backward_info['steps'])/300.0
check_signal_propagation.accuracy_requirement = 0.5


def check_basic_function(harness_class, train_dataset):
    # important: accuracy is checked by rounding output, and comparing the whole output vectors to each other
    # we could use a real loss function but that makes life a lot messier with these really simple functions...

    output_size = len(train_dataset[0][1])
    input_size = len(train_dataset[0][0])

    harness = harness_class(input_size=input_size, output_size=output_size)

    iterations = 20 * len(train_dataset) * (output_size + input_size)
    harness.train(dataset=train_dataset, iterations=iterations)

    number_correct_examples = 0

    for input_data, ground_truth in train_dataset:
        harness_output, _ = harness.forward(input_data)
        int_ground_truth = [int(round(f)) for f in ground_truth]
        int_harness_output = [int(round(f)) for f in harness_output]

        if all(a==b for a, b in zip(int_ground_truth, int_harness_output)):
            number_correct_examples += 1

    return float(number_correct_examples) / len(train_dataset)


def check_add(harness_class):
    train_dataset = [
        ([0, 0] , [0]),
        ([0, 1] , [1]),
        ([1, 0] , [1]),
        ([1, 1] , [2]),
    ]
    return check_basic_function(harness_class, train_dataset)
check_add.accuracy_requirement = 0.5


def check_subtract(harness_class):
    train_dataset = [
        ([0, 0] , [0]),
        ([0, 1] , [-1]),
        ([1, 0] , [1]),
        ([1, 1] , [0]),
    ]
    return check_basic_function(harness_class, train_dataset)
check_subtract.accuracy_requirement = 0.5


def check_and(harness_class):
    train_dataset = [
        ([0, 0] , [0]),
        ([0, 1] , [0]),
        ([1, 0] , [0]),
        ([1, 1] , [1]),
    ]
    return check_basic_function(harness_class, train_dataset)
check_and.accuracy_requirement = 0.5


def check_or(harness_class):
    train_dataset = [
        ([0, 0] , [0]),
        ([0, 1] , [1]),
        ([1, 0] , [1]),
        ([1, 1] , [1]),
    ]
    return check_basic_function(harness_class, train_dataset)
check_or.accuracy_requirement = 0.5


def check_switch(harness_class):
    train_dataset = [
        ([0, 0] , [0, 0]),
        ([0, 1] , [1, 0]),
        ([1, 0] , [0, 1]),
        ([1, 1] , [1, 1]),
    ]
    return check_basic_function(harness_class, train_dataset)
check_switch.accuracy_requirement = 0.5


all_checks = [check_signal_propagation, check_add, check_subtract, check_and, check_or, check_switch]
all_checks_accuracy_requirements = [check.accuracy_requirement for check in all_checks]
