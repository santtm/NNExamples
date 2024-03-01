import numpy as np


def softplus(x: float) -> float:
    return np.log(1 + np.exp(x))

def predict(input: float) -> float:
    processed_input = input * FIRST_WEIGHTS + FIRST_BIASES

    hidden_layer = np.vectorize(softplus)(processed_input)

    output = np.sum(hidden_layer * SECOND_WEIGHTS) + LAST_BIAS

    return output


FIRST_WEIGHTS = np.array([-34.4, -2.52])
FIRST_BIASES = np.array([2.14, 1.29])
SECOND_WEIGHTS = np.array([-1.3, 2.28])
LAST_BIAS = -0.58


print(predict(0.5))