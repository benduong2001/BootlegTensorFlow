from math import e as E
from BTF_tensor_classes import * 

class ACTIVATION_FUNCTION_HOLDER:
    def __init__(self):
        pass

    def apply(self, x):
        pass

    def apply_drv(self, x):
        pass


class SIGMOID(ACTIVATION_FUNCTION_HOLDER):
    def __init__(self):
        super(SIGMOID, self).__init__()

    def apply(self, m):
        nm = []
        for v in m.vr:
            x = v[0]
            nv = (1) / (1 + ((E) ** (-x)))
            nm.append([nv])
        return Matrix(nm)

    def apply_drv(self, m):
        nm = []
        for v in m.vr:
            x = v[0]
            nv0 = (1) / (1 + ((E) ** (-x)))
            nv = nv0 * (1 - nv0)
            nm.append([nv])
        return Matrix(nm)


class SOFTMAX(ACTIVATION_FUNCTION_HOLDER):
    # before the cross_entropy cost function,- there is the softmax activation layer
    # and before the softmax activation layer- there is the logits layer
    def __init__(self):
        super(SOFTMAX, self).__init__()

    def apply(self, matrix):
        # column vector input
        summed_averager = 0
        for v in matrix.vr:
            x = v[0]
            summed_averager += E ** (x)
        softmax_column = []
        for v in matrix.vr:
            x = v[0]
            softmax_column.append([E ** (x) / summed_averager])
        total_a0 = sum([x[0] for x in softmax_column])
        if not (int(round(total_a0)) == 1):  # > 0.9 and total_a0 <= 1.0): # adds up to 1 or is close
            raise AssertionError(str(total_a0) + " not close to 1")
        return Matrix(softmax_column)

    def apply_drv(self, matrix):
        # matrix is the z vector
        # https://youtu.be/wiDy8bd3F4A?t=993
        # print("softmax chain rule")
        # "https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/"
        # x is z_n, an aggregated value BEFORE being softmaxed, or Exponentiated
        # list of preactivated aggregators must no longer be modified
        sidelength = len(matrix.vr)
        jacobian = [[0 for _ in range(sidelength)] for _ in range(sidelength)]
        applied = self.apply(matrix)
        for i in range(sidelength):
            for j in range(sidelength):
                if i == j:
                    jacobian[i][j] = applied[i][0] * (1 - applied[i][0])
                else:
                    jacobian[i][j] = -applied[i][0] * applied[j][0]
        return Matrix(jacobian)

