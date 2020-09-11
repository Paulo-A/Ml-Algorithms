import copy

class NGradientDescend:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def weight_calc(self, weights, grad_weight):
        update_weights = []
        for weight in weights:
            update_weights.append(copy.deepcopy(weight))
        for jweight in range(len(weights[0])):
            for iweight in range(len(weights)):
                update_weights[iweight][jweight] = update_weights[iweight][jweight] - \
                    self.learning_rate*grad_weight[iweight][0]
        return update_weights

    def bias_calc(self, biases, grad_bias):
        update_biases = []
        for bias in biases:
            update_biases.append(copy.deepcopy(bias))
        for jbias in range(len(biases[0])):
            for ibias in range(len(biases)):
                update_biases[ibias][jbias] = update_biases[ibias][jbias] - \
                    self.learning_rate*grad_bias
        return update_biases
