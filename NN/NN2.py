import numpy as np


def softplus(x: float) -> float:
    return np.log(1 + np.exp(x))


def ssr(predicted: np.array) -> float:
    return np.sum(np.square(OBSERVED - predicted))

def ssr_derivative(predicted: np.array) -> float:
    return -2 * np.sum(OBSERVED - predicted)


def predict(input: float) -> float:
    processed_input = input * FIRST_WEIGHTS + FIRST_BIASES

    hidden_layer = np.vectorize(softplus)(processed_input)

    output = np.sum(hidden_layer * SECOND_WEIGHTS) + b3

    return output

def gradient_descent(prediction: np.array) -> float:
    slope = ssr_derivative(prediction)
    step_size = slope * LEARNING_RATE

    return b3 - step_size


def main() -> None:
    global b3
    for _ in range(MAX_OPERATIONS):
        prediction = np.vectorize(predict)(INPUTS)
        print(f'Prediction: {prediction}\nSSR: {ssr(prediction)}\nb3: {b3}\n')
        
        # backpropagation
        b3 = gradient_descent(prediction)


LEARNING_RATE = 0.1
MAX_OPERATIONS = 5

FIRST_WEIGHTS = np.array([3.34, -3.53])
FIRST_BIASES = np.array([-1.43, 0.57])
SECOND_WEIGHTS = np.array([-1.22, -2.3])
b3 = 0

INPUTS = [0, 0.5, 1]
OBSERVED = [0, 1, 0]


main()

