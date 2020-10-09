from BTF_tensor_classes import *
from BTF_data_inputs import Observation
import pandas as pd

class ENCODER_HOLDER:
    # NOT FOR DERIVATIVES
    # returns one_hot_encoded column matrices,
    # not the function's value itself
    def __init__(self):
        pass

    def apply(self, varbs):
        pass

class LINREG_ENCODER(ENCODER_HOLDER):
    def __init__(self):
        super(LINREG_ENCODER, self).__init__()
        self.table = [] # pd.DataFrame(columns = ["x", "y", "region"])
    def generate_dataset(self):
        observations = []
        for y in range(0, 100):
            for x in range(0, 100):
                # group = grouping(i, j) # group is onehot encoded column vector matrix
                group = self.apply([x, y])
                observation = Observation(Matrix([[x], [y]]), group)
                observations.append(observation)
        return observations
    def table_append_prediction(self, original, predicted):
        trial_x, trial_y = original.unwrap()
        dictrow = {"x": trial_x, "y": trial_y, "region": predicted}
        self.table.append(dictrow)

    def table_show(self):
        df = pd.DataFrame(self.table)
        fig = df.plot.scatter(x='x',
                              y='y',
                              c='region',
                              colormap='viridis')
    def apply(self, varbs):
        # since a0 for linreg loss is scalar, this returns scalars
        x = varbs[0]
        y = varbs[1]
        line = (1 / 3 * x) + 6
        if y <= line:
            # underneath
            return Matrix([[0]])
        else:
            return Matrix([[1]])

K = 20
class REGION_ENCODER(ENCODER_HOLDER):
    def __init__(self):
        super(REGION_ENCODER).__init__();
        self.table = [] # pd.DataFrame(columns = ["x", "y", "region"])
    def generate_dataset(self):
        observations = []
        for y in range(0, 100):
            for x in range(0 - K, 100 - K):
                # group = grouping(i, j) # group is onehot encoded column vector matrix
                group = self.apply([x, y])
                observation = Observation(Matrix([[x], [y]]), group)
                observations.append(observation)
        return observations
    def table_append_prediction(self, original, predicted):
        trial_x, trial_y = original.unwrap()
        dictrow = {"x": trial_x, "y": trial_y, "region": predicted}
        self.table.append(dictrow)

    def table_show(self):
        df = pd.DataFrame(self.table)
        fig = df.plot.scatter(x='x',
                              y='y',
                              c='region',
                              colormap='viridis')

    def apply(self, varbs):
        assert type(varbs) == list  # of ints
        assert len(varbs) == 2
        x = varbs[0]
        y = varbs[1]
        if x <= (20 - K):
            return Matrix([[1], [0], [0]])
        else:
            if x <= (45 - K):
                return Matrix([[0], [1], [0]])
            else:
                return Matrix([[0], [0], [1]])


class TEST_VERIFIER_HOLDER:  # FOR DEBUGGING PURPOSES ONLY
    # function holder for a function that
    # verifies whether the loss stage of an observation from the test dataset
    # is correct or not, to record successes in verify_test_dataset of categ_nn_executor

    def __init__(self):
        pass

    def apply(self, final_loss):
        # returns boolean
        pass


class DISTRIBUTION_VERIFIER(TEST_VERIFIER_HOLDER):
    def __init__(self):
        super(DISTRIBUTION_VERIFIER, self);

    def apply(self, final_loss):
        # assert type(final_loss) == Loss
        pred_dist = [abs(x) for x in final_loss.a0.unwrap()]
        targ_dist = [abs(x) for x in final_loss.freq_target.unwrap()]
        # print(pred_dist, targ_dist)
        targ_max_idx = max(range(len(targ_dist)), key=targ_dist.__getitem__)
        pred_max_idx = max(range(len(pred_dist)), key=pred_dist.__getitem__)
        max_idx_match_bool = targ_max_idx == pred_max_idx
        return max_idx_match_bool  # boolean


class LINREG_VERIFIER(TEST_VERIFIER_HOLDER):
    def __init__(self):
        super(LINREG_VERIFIER, self).__init__();

    def apply(self, final_loss):
        # assert type(final_loss) == Loss
        # expecting matrices
        pred_dist = final_loss.a0.unwrap()  # distance, not distribution
        targ_dist = final_loss.freq_target.unwrap()
        targ_label = targ_dist[0]  #
        if pred_dist[0] < 0.5:
            pred_label = 0
        else:
            pred_label = 1
        label_match_bool = targ_label == pred_label
        return label_match_bool  # boolean
