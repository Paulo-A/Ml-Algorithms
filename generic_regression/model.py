import copy

class LinearModel():
    def __init__(self, values, inital_weights, initial_biases):

        if type(inital_weights[0]) != list:
            self.weights = [copy.deepcopy(inital_weights)]
        else:
            self.weights = copy.deepcopy(inital_weights)

        if type(initial_biases[0]) != list:
            self.biases = [copy.deepcopy(initial_biases)]
        else:
            self.biases = copy.deepcopy(initial_biases)

        if type(values[0])!=list:
            self.values = [copy.deepcopy(values)]
        else:
            self.values = copy.deepcopy(values)

        self.__verify_dimensions__()

    def __values_weight_mult__(self):
        result_multi = []

        for ivalue in range(len(self.values)):
            result_multi.append([0]*len(self.weights[0]))

        for ivalue in range(len(self.values)):
            for jweight in range(len(self.weights[0])):
                for iweight in range(len(self.weights)):
                    result_multi[ivalue][jweight] += (
                        self.values[ivalue][iweight] *
                        self.weights[iweight][jweight]
                    )

        return result_multi

    def __verify_dimensions__(self):
        if (
            len(self.values[0]) == len(self.weights) and
            len(self.values) == len(self.biases) and
            1 == len(self.biases[0])
        ):
            pass
        elif len(self.values[0]) != len(self.weights):
            raise Exception(
                'Points and Weight matrix must have dimensions as dim(P)=[n,m] and dim(W)=[m,l]'
            )
        else:
            raise Exception(
                'Points, Weight and Bias matrix must have dimensions as dim(P)=[n,m], '+
                'dim(W)=[m,l] and dim(B)=[n,1]'
            )

    def compute(self):
        compute_result = self.__values_weight_mult__()
        for imult in range(len(compute_result)):
            for jmult in range(len(compute_result[0])):
                compute_result[imult][jmult] += self.biases[imult][0]
        return compute_result

    def update_model(self, weights, biases):
        self.weights = weights
        self.biases = biases
