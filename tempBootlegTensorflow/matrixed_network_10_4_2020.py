import random


import matplotlib.pyplot as plt

from BTF_data_inputs import *
from BTF_cost_function_holders import *
from BTF_activation_holders import *
from BTF_tensor_classes import *
from BTF_encoders_verifiers import *

class Layer:
    def __init__(self, layer_rank=0, network=None,
                 weights=None, biases=None,
                 inputs=None, activation=None):
        self.layer_rank = layer_rank

        network.all_layers.append(self)
        self.network = network

        self.W = weights
        self.network.all_parameters.append(self.W)
        self.b = biases
        self.network.all_parameters.append(self.b)
        self.a0 = inputs
        self.activation = activation
        # after self.forward is ran
        self.a1 = None
        self.next_layer = None

    def _concat_drv_a(self, drv_a, N):
        # creates matrix where column vector drv_a is repeated n # of times
        # where n = width of new matrix, but can be of a0 or a2's length, hypothetically

        # for drv_a, drv_a is the curr aN, and N is the len of drv_aN+1
        concat_array = [[vector_row[0] for _ in range(N)] for vector_row in drv_a.vr]
        concat_matrix = Matrix(concat_array)
        return concat_matrix

    def forward(self):
        W = self.W
        a0 = self.a0
        b = self.b
        activation = self.activation


        ws = (W * (a0))  # ws = weighted_sum
        z = ws + (b)
        a1 = activation.apply(z)
        return a1

    def drv_a0(self):
        # Deriving the front a, or "a0"
        # not a1
        # for this is on the right half that comes after the derivation of activation z
        # no need to worry about whether the next layer is loss
        a0 = self.a0
        nl = self.next_layer
        W = self.W
        b = self.b
        activation = self.activation

        next_drv_a = self.drv_a()

        
        scrwm = self.W.transpose()
        # print('scrwm\n'+str(scrwm)+'\n')

        drv_a0 = (scrwm) * (next_drv_a)  # chain rule vector

        
        # drv_a0 = a0.hadamard(chr_a1)

        return drv_a0
        

    def drv_a(self):
        # print("   MMMMMMM   ")
        # for a1 ( next layer's a0)
        # outputs a vector
        a0 = self.a0 # simply to create the drv_z1, not included in chain rule
        # in case you are confused with drv_front_a
        nl = self.next_layer
        W = self.W
        b = self.b
        activation = self.activation

        # is strictly for a1, not a0

        # if next not loss

        z1 = (W * (a0)) + (b)

        if type(activation) == SOFTMAX:
            jacobian = activation.apply_drv(z1)
            assert type(nl) == Loss
            assert type(nl.loss_function) == CROSSENTROPY
            loss_drv = nl.drv_loss()  # drv_column
            drv_z1 = jacobian * loss_drv  # matmul
            return drv_z1

        drv_z1 = self.activation.apply_drv(z1)
        # print('drv_z1\n'+str(drv_z1)+'\n')

        if type(nl) == Loss:  # BRANCH
            # if next layer is loss, this layer is output_activation
            next_drv_a = Matrix([[nl.drv_loss()[0][0]] for _ in range(len(drv_z1))])
            return drv_z1.hadamard(next_drv_a)
        else:
            next_drv_a = nl.drv_a()  # drv_a2
        # print('next_drv_a\n'+str(next_drv_a)+'\n')
        # a1 = nl.a0
        # print('a1\n'+str(a1)+'\n')

        # scrwm means sum_chain_rule_weight_matrix of Matrix(scrwa)
        # scrwm = self._concat_drv_a(a1, N = len(next_drv_a))

        scrwm = nl.W.transpose()
        # print('scrwm\n'+str(scrwm)+'\n')

        chr_a1 = (scrwm) * (next_drv_a)  # chain rule vector
        # print('chr_a1\n'+str(chr_a1)+'\n')

        drv_a1 = drv_z1.hadamard(chr_a1)
        # print('drv_a1\n'+str(drv_a1)+'\n')
        # print(" wwwwwww ")

        return drv_a1  # is vectorlike

    def drv_W(self, *W):
        # outputs a matrix
        W = self.W
        a0 = self.a0
        nl = self.next_layer
        a1 = self.next_layer.a0

        w_height = len(W)  # height of weight matrix
        dW_array = [a0.to_vector() for i in range(w_height)]  #
        dW = Matrix(dW_array)
        # print('dW\n'+str(dW)+'\n')
        next_drv_a = self.drv_a()

        dW_width = len(dW.vr[0])  # row length
        nldm = self._concat_drv_a(next_drv_a, dW_width)
        # print('nldm\n'+str(nldm)+'\n')
        # nldm - next layer derivative matrix
        # next layer vector (Drvs) or drv_a, repeated side by side to match weight matrix
        drv_W_ = dW.hadamard(nldm)
        return drv_W_  # is matrix

    def drv_b(self):
        # drv_b_ = self.drv_a() # WRONG, we must do it to the next layer
        drv_b_ = self.drv_a()
        return drv_b_  # is vectorlike

    def __repr__(self):
        str_form = """L-{0};\nA0-{1};\nB-{2};\nW-{3}\n"""
        return str_form.format(self.layer_rank, self.a0, self.b, self.W)

    def backpropagation(self):
        weights_grad = self.drv_W()
        biases_grad = self.drv_b()
        layer_gradients = [weights_grad, biases_grad]
        return layer_gradients

    def post_creation_organize(self, next_W=None, next_b=None, next_activation=None):
        # organizes everything after when the object is first created
        # OR When the layer is being updated
        # network and layer_rank are the only attributes that must be stated when the new layer is created
        # but we must assume that a0 is given

        assert self.network != None  # in order of priority
        assert self.a0 != None
        assert self.W != None
        assert self.b != None
        assert self.activation != None

        # WARNING: forward() is ran in this function

        assert self.a1 != None and self.next_layer != None
        nl = self.next_layer
        new_a1 = self.forward()  # with new a0 and reconfigured W and b
        self.a1.pointer_preserved_change(new_a1)
        nl.a0.pointer_preserved_change(new_a1)

    # THIS FUNCTION BELOW IS OUTDATED, DO NOT USE post_creation_organize_output
    def post_creation_organize_output(self, next_target=None, next_loss_function=None):
        # FOR PENULTIMATE LAYER
        # CASE 1: the object is first created:
        if self.a1 == None or self.next_layer == None:

            # WARNING: forward() is ran in this function

            # next_W, next_b, next_activation for a1 must be set MANUALLY
            # either by the function's arguments or after THIS function is ran
            a1 = self.forward()
            nl = Loss(a0=self.a0, network=self,
                      freq_target=next_target,
                      loss_function=next_loss_function)
            self.a1 = a1
            self.next_layer = nl
            ## if this was for a regular layer, we would have just used
            ##
            # a1 = self.forward()
            # nl  = Layer(layer_rank = self.layer_rank + 1, network = Onion,
            #                    weights = next_W, biases = next_b,
            #                    inputs = a1, activation = next_activation)
            # self.a1 = a1
            # self.next_layer = nl
            ## where next_W, next_b and next_activation are parameters = None

        # CASE 2: the object is being updated:
        # please know that: a0 has already been updated recursively and AUTOMATICALLY
        # and only case where a0 isn't updated is the first layer, or inputs
        # next_target, next_loss_function won't be used here since they were already reconfigured
        # but the next layer's a0 WILL BE modified (whiel preserving pointers)
        else:
            # here, the next layer exists already
            nl = self.next_layer
            new_a1 = self.forward()  # with new a0 and reconfigured W and b
            self.a1.pointer_preserved_change(new_a1)
            nl.a0.pointer_preserved_change(new_a1)


class Loss:
    def __init__(self, a0=None, network=None, freq_target=None, loss_function=None):
        self.a0 = self.output = a0  # post activation outputs before loss value
        network.final_loss = self
        self.network = network
        self.freq_target = freq_target  # unmatrixed array
        self.loss_function = loss_function
        self.a1 = self.loss = None

    def forward(self, p=None, t=None):
        if p == None:
            p = self.a0
        if t == None:
            assert self.freq_target != None
            t = self.freq_target
        loss = self.loss_function.apply(p, t)  # returns a matrix
        return loss

    def drv_loss(self, p=None, t=None):
        # p, the prediction matrix
        
        if p == None:
            p = self.a0
        if t == None:
            assert self.freq_target != None
            t = self.freq_target
        drv_loss = self.loss_function.apply_drv(p, t)
        return drv_loss

    def drv_a(self):
        p = self.a0
        pl = self.network.all_layers[-1]
        drv_z0 = pl.activation.apply_drv((pl.W * pl.a0) + pl.b)
        assert self.freq_target != None
        t = self.freq_target
        drv_loss = self.loss_function.apply_drv(p, t)  # returns a matrix
        return drv_z0.hadamard(drv_loss)  # is a matrix

    def post_creation_organize(self, freq_target=None):
        # print("ao is {0}, \n freq_target is {1}, \n a1 is {2}".format(self.a0, self.freq_target,self.a1))
        if freq_target == None:
            freq_target = self.freq_target
        # target may be changed ... outside
        # case 1: first time
        if self.loss == None or self.a1 == None:
            # given: target, a0, loss_function, network
            a1 = self.forward(self.a0, freq_target)
            self.a1 = self.loss = a1
        else:
            new_a1 = self.forward(self.a0, freq_target)
            self.a1.pointer_preserved_change(new_a1)
            self.loss.pointer_preserved_change(new_a1)

    def __repr__(self):
        str_form = """Loss;\nA0-{0};\ntarget-{1};\na1-{2}\n"""
        return str_form.format(self.a0, self.freq_target, self.a1)


class Categ_NN:
    def __init__(self):
        self.learning_rate = 0.5
        self.all_layers = []
        self.all_parameters = []
        self.final_loss = None

    def GD1_create_gradient(self):
        gradient = []
        for l_i in range(len(self.all_layers)):
            temp_layer = self.all_layers[l_i]
            temp_layer_gradient = temp_layer.backpropagation()
            temp_layer_weight_grad = temp_layer_gradient[0]
            gradient.append(temp_layer_weight_grad)
            temp_layer_bias_grad = temp_layer_gradient[1]
            gradient.append(temp_layer_bias_grad)
        assert len(gradient) == len(self.all_parameters)
        return gradient

    def GD2_finalise_gradient(self, gradient):
        # subtracts, learning rate, for reconfiguration method
        learning_gradient = [(part_drv * self.learning_rate) for part_drv in gradient]
        new_parameters = []
        for i in range(len(self.all_parameters)):
            parameter = self.all_parameters[i]
            part_drv = learning_gradient[i]
            difference = parameter - part_drv
            new_parameters.append(difference)
        assert len(new_parameters) == len(self.all_parameters)
        return new_parameters

    def GD3_reconfigure_parameters(self, new_parameters):
        # list of matrices, weight and bias (alternating)
        assert len(new_parameters) == len(self.all_parameters)
        for i in range(len(self.all_parameters)):
            new_parameter = new_parameters[i]
            self.all_parameters[i].pointer_preserved_change(new_parameter)

    def GD4_update_reconfigurations(self):
        for temp_layer in self.all_layers:
            temp_layer.post_creation_organize()
        self.final_loss.post_creation_organize()

    def reconfigure_observation(self, input_, target):
        # updates the input and output
        self.all_layers[0].a0.pointer_preserved_change(input_)
        self.final_loss.freq_target.pointer_preserved_change(target)

    # FUNCTIONS FOR TRAINING
    def run_batches(self, train_datalist, num_batches, network_obj=None):
        if network_obj == None:
            network_obj = self
        # network_obj with layers and loss already set to default values
        # train_datalist =  list of obsverations objects, scrambled
        # num_batches = int number of batch runs you want
        batch_len = len(train_datalist) // num_batches  # length (num of observations) per batch
        losses = []

        batch_run_i = 0
        imported_parameters = network_obj._rand_parameters(network_obj.all_parameters)
        # print(imported_parameters)
        while batch_run_i < num_batches:
            start = (batch_len * batch_run_i)
            end = (batch_len * batch_run_i) + batch_len
            train_sublist = train_datalist[start:end]

            exported_parameters = self._average_gradient(batch_observations=train_sublist,
                                                         imported_parameters=imported_parameters,
                                                         network_obj=None)
            imported_parameters = self.GD2_finalise_gradient(exported_parameters)
            batch_loss = self.final_loss.a1.__sum__() # UNDO 7/25/2020
            losses.append(batch_loss)
            print("BATCH {0}/{1} <= {2}".format(batch_run_i, num_batches, batch_loss))
            # print(imported_parameters)
            batch_run_i += 1
        plt.plot(losses)
        plt.show()
        print(imported_parameters)
        return imported_parameters

    def _average_gradient(self, batch_observations=None, imported_parameters=None, network_obj=None):
        assert batch_observations != None
        assert imported_parameters != None
        if network_obj == None:
            network_obj = self
        # bacth_observations list of observations
        # imported_parameters can be random parameters, or parameters of prev round AFTER FINALIZATION
        batch_n = len(batch_observations)

        # STEP 1: set up with gradient of first observation in batch
        starting_observation = batch_observations[0]
        starting_input_ = starting_observation.info
        starting_target = starting_observation.target

        network_obj.reconfigure_observation(starting_input_, starting_target)
        network_obj.GD3_reconfigure_parameters(imported_parameters)
        network_obj.GD4_update_reconfigurations()

        batch_parameters = network_obj.GD1_create_gradient()
        # first parameter in batch; all other parameter gradient in batch will be added to this
        # , then entrywise division of everything by # of observations in batch
        list_gradients = [batch_parameters]

        # STEP 2: summing gradients in batch
        batch_i = 1  # start from second index
        while batch_i < batch_n:

            # change inputs, target
            temp_observation = batch_observations[batch_i]
            input_ = temp_observation.info
            target = temp_observation.target
            network_obj.reconfigure_observation(input_, target)

            # paremeters need not be changed, already updated in the first batch observation
            # and each osbervation of batch is recycling the same network obj
            network_obj.GD4_update_reconfigurations()
            # print(str(network_obj.final_loss.a1) + " ---------{0}/{1}".format(batch_i, batch_n))
            temp_grad = network_obj.GD1_create_gradient()

            # collect gradient # aka adding to batch_parameters
            list_gradients.append(temp_grad)
            for j in range(len(temp_grad)):
                temp_part_drv = temp_grad[j]
                summed_part_drv = batch_parameters[j] + temp_part_drv
                batch_parameters[j] = summed_part_drv

            batch_i += 1
        # STEP 3: dividing each part_drv in gradient by batch_n
        inverse_summed_averager = 1 / (batch_n)
        for i in range(len(batch_parameters)):
            part_drv = batch_parameters[i]
            batch_parameters[i] = part_drv * inverse_summed_averager
        # STEP 4: finalize gradient for export in next batch round
        exported_parameters = (batch_parameters)
        return exported_parameters  # will be imported_parameters for next round

    def _rand_parameters(self, parameter_list):
        # creates a list of randomized matrices of floats
        # for each matrix in the parameter_lists, matching in terms of height and width

        # This is only used once in the beginning, because the first observation in the training dataset
        # will have parameters where the weight matrices and bias matrices are all just matrices of 1,
        # as placeholder, but then we will randomize it after teh second observation
        
        rand_parameter_list = []
        for parameter in parameter_list:
            assert type(parameter) == Matrix
            height = len(parameter.vr)
            width = len(parameter.vr[0].headc)
            rand_parameter_array = [[random.random() for _ in range(width)]
                                    for _ in range(height)]
            rand_parameter_matrix = Matrix(rand_parameter_array)
            rand_parameter_list.append(rand_parameter_matrix)
        return rand_parameter_list
    def set_layer_list(self):
        self.layer_list = []
    def append_layer(self, weight_matrix_shape, bias_matrix_shape, activation_function, input_matrix = None):
        # weight_matrix_shape is a tuple of 2 positive ints representing the dimensions of the default weight matrix,
        # in the order of height, width.
        # same thing goes for bias_matrix_shape
        # the input_matrix is only mandatory for the first layer, but beyond that, the most recent layer's a1 matrix will be used
        if input_matrix == None:
            assert self.layer_list != []
            input_matrix = self.layer_list[-1].a1
        assert type(weight_matrix_shape) == tuple
        assert weight_matrix_shape[0] >= 0, weight_matrix_shape[1] >= 0
        assert type(bias_matrix_shape) == tuple
        assert bias_matrix_shape[0] >= 0, bias_matrix_shape[1] >= 0
        assert len(weight_matrix_shape) == 2
        assert len(bias_matrix_shape) == 2
        rand_bheight, rand_bwidth = bias_matrix_shape
        bias_matrix = Matrix([[random.random() for _ in range(rand_bwidth)]
                                      for _ in range(rand_bheight)])
        
        assert type(input_matrix) == Matrix
        assert type(bias_matrix) == Matrix
        assert weight_matrix_shape[1] == len(input_matrix)
        # verifying weight_matrix's width matches input_matrix's height
        assert weight_matrix_shape[0] == len(bias_matrix)
        # verifying weight_matrix's height matches bias_matrix's height
        assert len(input_matrix[0].headc) == len(bias_matrix)
        # verifying input_matrix's width matches bias_matrix's width
        assert type(activation_function) == ACTIVATION_FUNCTION_HOLDER
        rand_height, rand_width = weight_matrix_shape
        rand_weight_matrix = Matrix([[random.random() for _ in range(rand_width)]
                                      for _ in range(rand_height)])
        l0 = Layer(layer_rank=len(self.layer_list),
                   network=self,
                   weights=rand_weight_matrix,
                   biases=bias_matrix,
                   inputs=input_matrix,
                   activation=activation_function
                   )
        if self.layer_list != []:
            self.layer_list[-1].next_layer = a1
        a1 = l0.forward()
        l0.a1 = a1
        self.layer_list.append(l0)



class Categ_NN_Executor:
    def __init__(self, categ_nn_obj, encoder_holder):
        self.categ_nn_obj = categ_nn_obj
        self.encoder_holder = encoder_holder
        self.set_test_verifier_holder(None)
        self.set_dataset([])
        self.train_dataset = []
        self.test_dataset = []

    def set_dataset(self, dataset):
        assert type(dataset) == list
        # assert type(dataset[0]) == Observation
        self.dataset = dataset

    def set_test_verifier_holder(self, test_verifier_holder):
        self.test_verifier_holder = test_verifier_holder

    def generate_dataset(self):
        observations = self.encoder_holder.generate_dataset()
        self.set_dataset(observations)
        return observations

    def split_dataset(self, train_proportion=None):
        if train_proportion == None:
            train_proportion = 0.8
        random.shuffle(self.dataset)

        dataset_len = len(self.dataset)
        split_index = int(dataset_len * 0.8)
        train_dataset = (self.dataset)[:split_index]
        self.train_dataset = train_dataset
        self.test_dataset = (self.dataset)[split_index:]

    def batch_gradient_descent(self, num_batches = 100):
        # returns optimized paramaters
        descended_parameters = (self.categ_nn_obj).run_batches(
            train_datalist=self.train_dataset,
            num_batches=num_batches)
        return descended_parameters

    def verify_test_dataset(self, final_parameters):
        assert self.test_verifier_holder != None
        self.categ_nn_obj.GD3_reconfigure_parameters(final_parameters)
        successes = 0

        for i in range(len(self.test_dataset)):            
            trial = self.test_dataset[i]
            trial_input_ = trial.info
            trial_target = trial.target
            self.categ_nn_obj.reconfigure_observation(trial_input_, trial_target)
            # print("old input a0 {0}".format(str(self.categ_nn_obj.all_layers[0].a0)))
            # print("old loss a0 {0}".format(str(self.categ_nn_obj.final_loss.a0)))
            # print("old target a0 {0}".format(str(self.categ_nn_obj.final_loss.freq_target)))
            self.categ_nn_obj.GD4_update_reconfigurations()
            loss_stage = self.categ_nn_obj.final_loss
            if type(self.encoder_holder) == LINREG_ENCODER:
                loss_a0 = loss_stage.a0.unwrap()
                trial_type = int(loss_a0[0] <= 0.5)
                self.encoder_holder.table_append_prediction(trial_input_, trial_type)
            if type(self.encoder_holder) == REGION_ENCODER:
                loss_a0 = loss_stage.a0.unwrap()
                trial_type = max(range(len(loss_a0)), key=loss_a0.__getitem__)
                self.encoder_holder.table_append_prediction(trial_input_, trial_type)
            # print("new input a0 {0}".format(str(self.categ_nn_obj.all_layers[0].a0)))
            # print("new loss a0 {0}".format(str(self.categ_nn_obj.final_loss.a0)))
            # print("new target a0 {0}".format(str(self.categ_nn_obj.final_loss.freq_target)))
            # assert str(loss_stage.freq_target) == str(trial_target)
            verifier = self.test_verifier_holder.apply(loss_stage)
            if verifier == True:
                successes += 1
        print(str(successes) + "/" + str(len(self.test_dataset)))

        self.encoder_holder.table_show()
        plt.show()

        accuracy = (successes / (len(self.test_dataset)))
        return accuracy

    def main_executor(self):
        self.generate_dataset()
        self.split_dataset()
        final_parameters = self.batch_gradient_descent()
        accuracy = self.verify_test_dataset(final_parameters)
        return accuracy
