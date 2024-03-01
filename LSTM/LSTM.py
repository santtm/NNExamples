import numpy as np
from typing import Callable


def sigmoid(x: float) -> float:
    return np.exp(x)/(np.exp(x) + 1)

def tanh(x: float) -> float:
    return np.tanh(x)


class LstmCell:

    def generic_node(self, weights: np.array, bias: float, activation_function: Callable) -> float:
        processed_input = np.sum(self.inputs * weights) + bias
        hidden_layer = activation_function(processed_input)
        return hidden_layer


    def forget_gate(self) -> float:
        p_long_term_to_remember = self.generic_node(FORGET_GATE_WEIGHTS,
                                                    FORGET_GATE_BIAS,
                                                    sigmoid)

        self.long_term_memory *= p_long_term_to_remember
        
        # print(f'{p_long_term_to_remember = :.4f}')
        # print(f'{self.long_term_memory = :.4f}')

    def input_gate(self) -> float:
        p_potential_long_memory_to_remember = self.generic_node(INPUT_GATE_WEIGHTS[0],
                                                                INPUT_GATE_BIAS[0],
                                                                sigmoid)

        potential_long_term_memory = self.generic_node(INPUT_GATE_WEIGHTS[1],
                                                       INPUT_GATE_BIAS[1],
                                                       tanh)

        new_long_term_memory = self.long_term_memory + p_potential_long_memory_to_remember * potential_long_term_memory
        self.long_term_memory = new_long_term_memory

        # print(f'{p_potential_long_memory_to_remember = :.4f}')
        # print(f'{potential_long_term_memory = :.4f}')        
        # print(f'{new_long_term_memory = :.4f}')

    def output_gate(self) -> float:
        p_potential_short_memory_to_remember = self.generic_node(OUTPUT_GATE_WEIGHTS,
                                                                 OUTPUT_GATE_BIAS,
                                                                 sigmoid)
        
        potential_short_term_memory = tanh(self.long_term_memory)

        new_short_term_memory = p_potential_short_memory_to_remember * potential_short_term_memory
        self.short_term_memory = new_short_term_memory

        # print(f'{p_potential_short_memory_to_remember = :.4f}')
        # print(f'{potential_short_term_memory = :.4f}')
        # print(f'{new_short_term_memory = :.4f}')


    def compute(self, long_term_memory: float, short_term_memory: float, input: float) -> None:
        self.inputs = np.array([short_term_memory, input])
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory

        self.forget_gate()
        self.input_gate()
        self.output_gate()
    
    def get_lstm(self) -> tuple[float]:
        return self.long_term_memory, self.short_term_memory
    

def main():

    lstm = LstmCell()

    long_term_memory = 0
    short_term_memory = 0

    print('Company A\n')
    for input in [0, 0.5, 0.25, 1]:
        lstm.compute(long_term_memory, short_term_memory, input)
        long_term_memory, short_term_memory = lstm.get_lstm()
        
        print(f'Input: {input}')
        print(f'Long term memory: {long_term_memory}, Short term memory: {short_term_memory}\n')

    long_term_memory = 0
    short_term_memory = 0

    print('\nCompany B\n')
    for input in [1, 0.5, 0.25, 1]:
        lstm.compute(long_term_memory, short_term_memory, input)
        long_term_memory, short_term_memory = lstm.get_lstm()
        
        print(f'Input: {input}')
        print(f'Long term memory: {long_term_memory}, Short term memory: {short_term_memory}\n')


FORGET_GATE_WEIGHTS = np.array([2.7, 1.63])
FORGET_GATE_BIAS = 1.62

INPUT_GATE_WEIGHTS = np.array([[2, 1.65], [1.41, 0.94]])
INPUT_GATE_BIAS = np.array([0.62, -0.32])

OUTPUT_GATE_WEIGHTS = np.array([4.38, -0.19])
OUTPUT_GATE_BIAS = 0.59


main()

