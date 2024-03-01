import numpy as np
from typing import Callable


def softplus(x: float) -> float:
    return np.log(1 + np.exp(x))


def ssr(predicted: np.array) -> float:
    return np.sum(np.square(OBSERVED - predicted))

def ssr_derivative_b3(predicted: np.array) -> float:
    return -2 * np.sum(OBSERVED - predicted)

def ssr_derivative_w3(predicted: np.array) -> float:
    return -2 * np.sum((OBSERVED - predicted) * y_1i)

def ssr_derivative_w4(predicted: np.array) -> float:
    return -2 * np.sum((OBSERVED - predicted) * y_2i)


def predict(input: float) -> float:
    global y_1i, y_2i

    processed_input = input * FIRST_WEIGHTS + FIRST_BIASES

    '''
    If otypes is not specified, then a call to the function with the first argument
    will be used to determine the number of outputs. The results of this call will be cached
    if cache is True to prevent calling the function twice.
    (-_-)
    '''
    vectorized = np.vectorize(softplus, otypes=[float])
    hidden_layer = vectorized(processed_input)

    y_1i = np.append(y_1i, hidden_layer[0])
    y_2i = np.append(y_2i, hidden_layer[1])

    output = np.sum(hidden_layer * np.array([w3, w4])) + b3

    return output

def gradient_descent(target: float, prediction: np.array, derivative: Callable) -> float:
    slope = derivative(prediction)
    step_size = slope * LEARNING_RATE

    return target - step_size


def main() -> None:
    global w3, w4, b3
    global y_1i, y_2i

    for count in range(MAX_OPERATIONS):
        vectorized = np.vectorize(predict, otypes=[float])
        prediction = vectorized(INPUTS)

        if count % 100 == 0:
            print(f'Prediction: {prediction}\nSSR: {ssr(prediction)}\nw3:{w3} w4:{w4}\nb3:{b3}\n')
        
        # backpropagation
        b3 = gradient_descent(b3, prediction, ssr_derivative_b3)
        w3 = gradient_descent(w3, prediction, ssr_derivative_w3)
        w4 = gradient_descent(w4, prediction, ssr_derivative_w4)

        y_1i = np.array([])
        y_2i = np.array([])
    

LEARNING_RATE = 0.1
MAX_OPERATIONS = 500

FIRST_WEIGHTS = np.array([3.34, -3.53])
FIRST_BIASES = np.array([-1.43, 0.57])

# w3, w4 = np.random.randn(2)
w3, w4 = 0.36, 0.63
b3 = 0

y_1i = np.array([])
y_2i = np.array([])


INPUTS = [0, 0.5, 1]
OBSERVED = [0, 1, 0]


main()

