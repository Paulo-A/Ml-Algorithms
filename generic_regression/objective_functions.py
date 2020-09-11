class ModL2Norm():
    def loss_function(self, values, calculated, target):
        accuracy=0
        for icalculated in range(len(calculated)):
            accuracy += (calculated[icalculated][0] -
                         target[icalculated][0]) ** 2
        return accuracy/2/len(values)

    def grad_weight(self, values, calculated, target):
        grad_weight_sum = []
        while len(grad_weight_sum) < len(values[0]):
            grad_weight_sum.append([0])
        for jvalue in range(len(values[0])):
            for icalculated in range(len(calculated)):
                grad_weight_sum[jvalue][0] += (values[icalculated][jvalue] * \
                                            (calculated[icalculated][0]-target[icalculated][0])) / \
                                                len(values)
        return grad_weight_sum

    def grad_bias(self, values, calculated, target):
        grad_bias_sum = 0
        for icalculated in range(len(calculated)):
            grad_bias_sum += (calculated[icalculated]
                              [0]-target[icalculated][0])
        return grad_bias_sum/len(values)

