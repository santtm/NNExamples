import numpy as np
from inputs import *


def apply_filter(input: np.array, filter: np.array, bias: float) -> np.array:
    n_rows = len(input) - 2
    n_cols = len(input[0]) - 2
    
    feature_map = np.zeros((n_rows, n_cols))

    for row in range(n_rows):
        for col in range(n_cols):
            submatrix = input[row:row+3, col:col+3]
            dot_product = np.sum(filter * submatrix)
            feature_map[row, col] = dot_product + bias
            
    return feature_map        


def max_pooling(input: np.array) -> np.array:
    n_rows = len(input) - 2
    n_cols = len(input[0]) - 2
    
    max_pooled = np.zeros((n_rows, n_cols))
    
    for row in range(0, len(input), 2):
        for col in range(0, len(input), 2):
            submatrix = input[row:row+2, col:col+2]
            max_element = submatrix.max()
            max_pooled[row//n_rows, col//n_cols] = max_element
    
    return max_pooled

def mean_pooling(input: np.array) -> np.array:
    n_rows = len(input) - 2
    n_cols = len(input[0]) - 2
    
    mean_pooled = np.zeros((n_rows, n_cols))
    
    for row in range(0, len(input), 2):
        for col in range(0, len(input), 2):
            submatrix = input[row:row+2, col:col+2]
            mean_element = submatrix.mean()
            mean_pooled[row//n_rows, col//n_cols] = mean_element
    
    return mean_pooled


def argmax(input: np.array) -> np.array:
    max_element_idx = np.argmax(input)
    new_arr = np.zeros_like(input)
    new_arr[max_element_idx] = 1
    return new_arr

def softmax(input: np.array) -> np.array:
    exp_matrix = np.exp(input)
    sum_exp = np.sum(exp_matrix)
    return exp_matrix / sum_exp


def classification(input : np.array, pooling: str, argmax_or_softmax: str) -> str:
    feature_map = apply_filter(input, FILTER, FILTER_BIAS)

    # ReLU
    feature_map = np.maximum(feature_map, 0)

    if pooling == 'max':
        pooled = max_pooling(feature_map)
    elif pooling == 'mean':
        pooled = mean_pooling(feature_map)

    input_layer = pooled.flatten()

    processed_input = np.sum(input_layer * WEIGHTS) + INPUT_BIAS

    ReLu = np.maximum(processed_input, 0)

    output = [0, 0]
    output[0] = ReLu * X_WEIGHT_BIAS[0] + X_WEIGHT_BIAS[1]
    output[1] = ReLu * O_WEIGHT_BIAS[0] + O_WEIGHT_BIAS[1]

    output = argmax(output)

    prediction = 'X' * int(output[0]) + 'O' * int(output[1])

    if argmax_or_softmax == 'softmax':
        output = softmax(output)

    stroutput = f'Prediction: {prediction}\n{pooling.capitalize()}, {argmax_or_softmax.capitalize()}:\nX: {output[0]:.3f}, O: {output[1]:.3f}'

    return stroutput


def print_input(input: np.array) -> None:
    print(' _____________ ')
    for i in input:
        print('|', end=' ')
        for j in i:
            if j == 0:
                print(' ', end=' ')
            else:    
                print(u"\u25A0", end=' ')
        print('|')
    print(f' {"\u203e" * 13} ')


FILTER = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])

FILTER_BIAS = -2

WEIGHTS = np.array([-0.8, -0.07, 0.2, 0.17])

INPUT_BIAS = 0.97

X_WEIGHT_BIAS = [1.33, -0.45]
O_WEIGHT_BIAS = [-1.33, 1.45]


for input in INPUTS:
    var_name = [name for name, value in locals().items() if value is input][0]
    print(f'Input: {var_name}')
    print_input(input)
    print()
    print(classification(input, pooling='max', argmax_or_softmax='argmax'))
    print()
    print(classification(input, pooling='max', argmax_or_softmax='softmax'))
    print()
    print(classification(input, pooling='mean', argmax_or_softmax='argmax'))
    print()
    print(classification(input, pooling='mean', argmax_or_softmax='softmax'))
    print()
    print()

