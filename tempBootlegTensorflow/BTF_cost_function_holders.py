from math import log
from BTF_tensor_classes import * 


class COST_FUNCTION_HOLDER:
    def __init__(self):
        pass

    def apply(self, predicted_vector, target_vector):
        # To be overriden
        pass

    def apply_drv(self, predicted_vector, target_vector):
        # To be overriden
        pass


class MEAN_SQUARED_ERROR(COST_FUNCTION_HOLDER):
    def __init__(self):
        super(MEAN_SQUARED_ERROR, self).__init__()

    def apply(self, predicted_vector, target_vector):
        final_value = 0
        for i in range(len(predicted_vector)):
            error = (predicted_vector[i][0] - target_vector[i][0])
            squared_error = error * error
            final_value += squared_error
        return Matrix([[final_value]])

    def apply_drv(self, predicted_vector, target_vector):
        final_value = 0
        for i in range(len(predicted_vector)):
            error = (predicted_vector[i][0] - target_vector[i][0])
            drv_error = 2 * error
            final_value += drv_error
        return Matrix([[final_value]])


class CROSSENTROPY(COST_FUNCTION_HOLDER):
    def __init__(self):
        super(CROSSENTROPY, self).__init__()
        # be sure to set this before using the cost_function

    def apply(self, predicted_vector, target_vector):
        if type(predicted_vector) == Matrix:
            predicted_vector = predicted_vector.to_vector()
        if type(target_vector) == Matrix:
            target_vector = target_vector.to_vector()
        # lists are allowed too and don't need to be converted
        sum_log_loss_drv = []
        vector_dimen = len(predicted_vector)
        softmax_vector = predicted_vector

        for i in range(vector_dimen):
            p = softmax_vector[i]
            t = target_vector[i]
            temp_log_loss_drv = t * log(p + 1e-20)
            sum_log_loss_drv.append([temp_log_loss_drv])
        # REVISIT:
        # You will need the number of examples used, i.e. size of the training dataset
        # http://neuralnetworksanddeeplearning.com/chap3.html
        return Matrix(sum_log_loss_drv)

    def apply_drv(self, predicted_vector, target_vector):
        if type(predicted_vector) == Matrix:
            predicted_vector = predicted_vector.to_vector()
        if type(target_vector) == Matrix:
            target_vector = target_vector.to_vector()
        # lists are allowed too and don't need to be converted
        # https://www.youtube.com/watch?v=5-rVLSc2XdE
        sum_log_loss_drv = []
        vector_dimen = len(predicted_vector)
        softmax_vector = predicted_vector
        for i in range(vector_dimen):
            p = softmax_vector[i]
            t = target_vector[i]

            temp_log_loss_drv = (t * (1 / (p + 1e-8))) * (-1)
            sum_log_loss_drv.append([temp_log_loss_drv])
        return Matrix(sum_log_loss_drv)

