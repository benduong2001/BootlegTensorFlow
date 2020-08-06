from matrixed_network_7_25_2020 import *

from PIL import Image
# import os
# In[ ]:





# In[3]:


class Conv_nn:
    def __init__(self):
        self.learning_rate = 0.01
        self.all_layers = []
        self.all_parameters = []
        self.categ_parameters = []
        self.final_loss = None

    def GD1_create_gradient(self):
        gradient = []
        for l_i in range(len(self.all_layers)):
            temp_layer = self.all_layers[l_i]
            if type(temp_layer) == Conv_Layer:
                temp_layer_gradient = temp_layer.backpropagation()
                temp_layer_weight_grad = temp_layer_gradient[0]
                gradient.append(temp_layer_weight_grad)
                temp_layer_bias_grad = temp_layer_gradient[1]
                gradient.append(temp_layer_bias_grad)
        assert len(gradient) == len(self.all_parameters)
        assert type(gradient) == list
        assert type(gradient[0]) == list
        assert type(gradient[0][0]) == Tensor_3D
        return gradient

    def GD2_finalise_gradient(self, gradient):
        # subtracts, learning rate, for reconfiguration method
        assert type(gradient) == list
        assert type(gradient[0]) == list
        assert type(gradient[0][0]) == Tensor_3D
        learning_gradient = []
        # (part_drv * self.learning_rate) for part_drv in gradient]
        new_parameters = []
        for i in range(len(self.all_parameters)):
            new_parameter = []
            parameter = self.all_parameters[i]
            part_drv = gradient[i]
            for j in range(len(parameter)):
                
                tensor = parameter[j]
                new_tensor = part_drv[j]

                new_tensor = new_tensor * self.learning_rate
                
                diff_tensor = tensor - new_tensor
                new_parameter.append(diff_tensor)
            new_parameters.append(new_parameter)
        assert len(new_parameters) == len(self.all_parameters)
        assert type(new_parameters) == list
        assert type(new_parameters[0]) == list
        assert type(new_parameters[0][0]) == Tensor_3D
        return new_parameters

    def GD3_reconfigure_parameters(self, new_parameters):
        assert type(new_parameters) == list
        assert type(new_parameters[0]) == list
        assert type(new_parameters[0][0]) == Tensor_3D
        assert len(new_parameters) == len(self.all_parameters)

        for i in range(len(self.all_parameters)):
            new_parameter = new_parameters[i]
            for j in range(len(new_parameter)):

                new_tensor = new_parameter[j]
                self.all_parameters[i][j].pointer_preserved_change(new_tensor)
    def GD4_update_reconfigurations(self):
        for temp_layer in self.all_layers:
            temp_layer.post_creation_organize()


    def reconfigure_observation(self, input_, target):
        # updates the input and output

        self.all_layers[0].a0.pointer_preserved_change(input_)
        self.all_layers[-1].categ_nn.final_loss.freq_target.pointer_preserved_change(target)

    # FUNCTIONS FOR TRAINING
    def run_batches(self, train_datalist, num_batches, network_obj=None):


        # NOTES REGARDING FOR CATEG:
        # IN THIS FUNCTION, THE CATEG_NN USED IT'S OWN FUNCTION IN ALL CASES EXCEPT:
        # 1) _average_gradient, WHICH WAS COMBINED INTO THE CONV's AVERAGE_GRADIENT
        # 2) reconfigure_observation, WHERE CONV's VERSION ALREADY COVERED CATEG_NN
        # 3) GD4_update_configurations, WHERE CONV's VERSION ALREADY COVERED CATEG_NN
        if network_obj == None:
            network_obj = self
        categ_nn_obj = network_obj.all_layers[-1].categ_nn
        # network_obj with layers and loss already set to default values
        # train_datalist =  list of obsverations objects, scrambled
        # num_batches = int number of batch runs you want
        batch_len = len(train_datalist) // num_batches  # length (num of observations) per batch
        losses = []

        batch_run_i = 0
        imported_parameters = network_obj._rand_parameters(network_obj.all_parameters)
        # categ_nn_obj = network_obj.all_layers[-1].categ_nn
        imported_categ_parameters = categ_nn_obj._rand_parameters(categ_nn_obj.all_parameters) #FOR CATEG _RAND_PARAMETERS

        # print(imported_parameters)
        while batch_run_i < num_batches:
            start = (batch_len * batch_run_i)
            end = (batch_len * batch_run_i) + batch_len
            train_sublist = train_datalist[start:end]

            exported_parameters, exported_categ_parameters = self._average_gradient(batch_observations=train_sublist,
                                                         imported_parameters=imported_parameters,
                                                         imported_categ_parameters=imported_categ_parameters,
                                                         network_obj=None) # FOR CATEG (DUAL OUTPUT RETURNED)
            imported_parameters = self.GD2_finalise_gradient(exported_parameters) 
            imported_categ_parameters = categ_nn_obj.GD2_finalise_gradient(exported_categ_parameters) # FOR CATEG GD2
            batch_loss = self.all_layers[-1].categ_nn.final_loss.a1.__sum__() # UNDO 7/25/2020
            losses.append(batch_loss)
            print("BATCH {0}/{1} <= {2}".format(batch_run_i, num_batches, batch_loss))
            # print(imported_parameters)
            batch_run_i += 1
        plt.plot(losses)
        plt.show()
        print("conv parameters:")
        print(imported_parameters)
        print("categ parameters:")
        print(imported_categ_parameters)
        return imported_parameters, imported_categ_parameters

    def _average_gradient(self, batch_observations=None, imported_parameters=None,
                          imported_categ_parameters=None, network_obj=None):
        assert batch_observations != None
        assert imported_parameters != None
        assert imported_categ_parameters != None
        if network_obj == None:
            network_obj = self
        categ_nn_obj = network_obj.all_layers[-1].categ_nn
        # bacth_observations list of observations
        # imported_parameters can be random parameters, or parameters of prev round AFTER FINALIZATION
        batch_n = len(batch_observations)

        # STEP 1: set up with gradient of first observation in batch
        starting_observation = batch_observations[0]
        starting_input_ = starting_observation.info
        starting_target = starting_observation.target

        network_obj.reconfigure_observation(starting_input_, starting_target)

        assert type(imported_parameters) == list
        assert type(imported_parameters[0]) == list
        assert type(imported_parameters[0][0]) == Tensor_3D
        network_obj.GD3_reconfigure_parameters(imported_parameters)

        categ_nn_obj.GD3_reconfigure_parameters(imported_categ_parameters) # FOR CATEG GD3
        
        network_obj.GD4_update_reconfigurations()

        batch_parameters = network_obj.GD1_create_gradient()
        # first parameter in batch; all other parameter gradient in batch will be added to this
        # , then entrywise division of everything by # of observations in batch


        batch_categ_parameters = categ_nn_obj.GD1_create_gradient() # FOR CATEG GD1

        list_gradients = [batch_parameters]
        list_categ_gradients = [batch_categ_parameters]# FOR CATEG

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
            temp_categ_grad = categ_nn_obj.GD1_create_gradient() # FOR CATEG

            # collect gradient # aka adding to batch_parameters
            list_gradients.append(temp_grad)
            list_categ_gradients.append(temp_categ_grad) # FOR CATEG
            for i in range(len(temp_grad)):
                temp_part_drv = temp_grad[i] # new grad
                for j in range(len(temp_part_drv)):
                    temp_tensor = temp_part_drv[j]
                    summed_part_drv = batch_parameters[i][j] + temp_tensor
                    batch_parameters[i][j] = summed_part_drv


            # FOR CATEG
            for j in range(len(temp_categ_grad)): # FOR CATEG
                temp_categ_part_drv = temp_categ_grad[j] # FOR CATEG
                summed_categ_part_drv = batch_categ_parameters[j] + temp_categ_part_drv # FOR CATEG
                batch_categ_parameters[j] = summed_categ_part_drv # FOR CATEG
            batch_i += 1
            
        # STEP 3: dividing each part_drv in gradient by batch_n
        inverse_summed_averager = 1 / (batch_n)
        for i in range(len(batch_parameters)):
            part_drv = batch_parameters[i]
            for j in range(len(part_drv)): # each param is a list of tensors
                summed_tensor = part_drv[j]
                batch_parameters[i][j] = summed_tensor * inverse_summed_averager


        for i in range(len(batch_categ_parameters)): # FOR CATEG
            part_categ_drv = batch_categ_parameters[i] # FOR CATEG
            batch_categ_parameters[i] = part_categ_drv * inverse_summed_averager # FOR CATEG
        
        # STEP 4: finalize gradient for export in next batch round
        exported_parameters = (batch_parameters)
        exported_categ_parameters = batch_categ_parameters # FOR CATEG
        return exported_parameters, exported_categ_parameters  # FOR CATEG # will be imported_parameters for next round

    def _rand_parameters(self, parameter_list):
        assert type(parameter_list) == list
        assert type(parameter_list[0]) == list
        assert type(parameter_list[0][0]) == Tensor_3D
        rand_parameter_list = []
        for i in range(len(parameter_list)):
            assert type(parameter_list[i]) == list
            assert type(parameter_list[i][0]) == Tensor_3D
            new_tensors = []
            parameter = parameter_list[i]
            for tensor in parameter:
                height = tensor.height
                width = tensor.width
                depth = tensor.depth
                np_parameter = np.random.rand(depth, height, width).astype(float)
                new_tensor = Tensor_3D(np_parameter.tolist())
                new_tensors.append(new_tensor)
            rand_parameter_list.append(new_tensors)
        assert type(rand_parameter_list) == list
        assert type(rand_parameter_list[0]) == list
        assert type(rand_parameter_list[0][0]) == Tensor_3D
        return rand_parameter_list

        
        
class Cnn_Layer:
    def __init__(self, network = None, layer_rank = 0, a0 = None):
        
        network.all_layers.append(self)
        self.network = network;
        self.layer_rank = layer_rank
        self.a0 = a0
        self.next_layer = None
    def chain_rule_tensor(self, curr_drv_tensor):
        nl = self.next_layer
        if type(nl) == Pool_Layer:
            nl.prev_drv_shape = curr_drv_tensor
            return (curr_drv_tensor).hadamard(nl.drv_a0())
        if type(nl) == Fully_Connected_Layer:
            return curr_drv_tensor.hadamard(nl.drv_a0())        
        if type(nl) == Conv_Layer:
            return curr_drv_tensor.hadamard(nl.drv_a0())

class Conv_Layer (Cnn_Layer):
    def __init__(self, network = None, layer_rank = 0, a0 = None, 
                 Fs = [], Bs = [], stride = 1, pad = 0):
        super(Conv_Layer, self).__init__(network = network,
                                         layer_rank = layer_rank,
                                         a0 = a0)

        self.Fs = Fs
        self.network.all_parameters.append(self.Fs)
        self.Bs = Bs
        self.network.all_parameters.append(self.Bs)
        self.R = self.a1 = None
        self.stride = stride
        self.pad = pad
    
    def _filt_matrix(self, orig, filt, stride = None, pad = None):
        # orig and filt are matrix
        o_h = len(orig.vr) # height of orig
        f_h = len(filt.vr) # height of filter
        
        o_w = len(orig.vr[0].headc) # width of orig
        f_w = len(filt.vr[0].headc) # width of filter
        if stride == None:
            stride = self.stride
        if pad == None:
            pad = self.pad
        
        output_h = (((o_h + (2 * pad)) - f_h)//stride) + 1
        output_w = (((o_w + (2 * pad)) - f_w)//stride) + 1
        
        output_matrix = [[0 for _ in range(output_w)]
                         for _ in range(output_h)]

        for y in range(0, output_h):
            for x in range(0, output_w):
                y1 = y * stride
                x1 = x * stride
                y2 = y1 + f_h
                x2 = x1 + f_w
                submatrix = orig.sector(x1, x2 - 1, y1, y2 - 1)
                # print(submatrix)
                hadamarded = submatrix.hadamard(filt)
                weighted_sum = hadamarded.__sum__()
                output_matrix[y][x] = weighted_sum
        return Matrix(output_matrix)
    def forward_Fs(self, a0 = None, weight_tensors = None, stride = None, pad = None):
        if a0 == None:
            a0 = self.a0
        if weight_tensors == None:
            weight_tensors = self.Fs
        
        F_tensor = []
        for weight_tensor in weight_tensors:

            assert a0.depth == weight_tensor.depth
            F_matrix = None # Final matrix, which is component-wise Added
            for i in range(weight_tensor.depth):

                orig_matrix = a0.md[i]
                weight_matrix = weight_tensor.md[i]
                filt_layer_matrix = self._filt_matrix(orig_matrix, weight_matrix,
                                                     stride = stride, pad = pad)
                
                if i == 0: # at the start, set the F_matrix to the first layer
                    F_matrix = filt_layer_matrix
                else: # after that, do matrix addition
                    # print(F_matrix)
                    # print(filt_layer_matrix)
                    F_matrix = F_matrix + filt_layer_matrix
            F_tensor.append(F_matrix)
        F_tensor_obj = Tensor_3D(F_tensor)
        return F_tensor_obj
    def forward_Fs1(self, a0 = None, weight_tensors = None, stride = None, pad = None):
        if a0 == None:
            a0 = self.a0
        if weight_tensors == None:
            weight_tensors = self.Fs
        if stride == None:
            stride = self.stride
        if pad == None:
            pad = self.pad
        F_tensor = []
        
        o = a0
        f = weight_tensors[0]

        o_h, o_w = o.height, o.width
        f_h, f_w = f.height, f.width
        output_h = (((o_h + (2 * pad)) - f_h)//stride) + 1
        output_w = (((o_w + (2 * pad)) - f_w)//stride) + 1

        
        
        for i in range(len(weight_tensors)):
            weight_tensor = weight_tensors[i]
            print("fs {0}/{1}".format(str(i), str(len(weight_tensors))))
            assert a0.depth == weight_tensor.depth

            weight_arr = weight_tensor.raw_form()
            filt = np.array(weight_arr)
            a0_arr = a0.raw_form()
            orig = np.array(a0_arr)
            
            F_matrix = [[0.0 for _ in range(output_w)]
                         for _ in range(output_h)]

            for y in range(0, output_h):
                for x in range(0, output_w):
                    y1 = y * stride
                    x1 = x * stride
                    y2 = y1 + f_h
                    x2 = x1 + f_w
                    ix_grid = np.ix_(list(range(0, a0.depth)),
                                     list(range(y1, y2)),
                                     list(range(x1, x2)))
                    submatrix = orig[ix_grid]
                    F_matrix[y][x] = float(np.sum(submatrix * filt))
            F_tensor.append(F_matrix)
        F_tensor_obj = Tensor_3D(F_tensor)
        assert F_tensor_obj.depth == len(weight_tensors)
        assert F_tensor_obj.height == output_h
        assert F_tensor_obj.width == output_w
        return F_tensor_obj
    def forward_Bs(self, tensor):
        # print(tensor)
        # print(self.Bs)
        return tensor + self.Bs[0] # matrix addition
    def forward_R(self, tensor):
        R_tensor = []
        for matrix in tensor.md:
            R_matrix = []
            for vector in matrix.vr:
                R_vector = []
                for x in vector.headc:
                    if x > 0:
                        R_vector.append(x)
                    else:
                        R_vector.append(0)
                R_matrix.append(R_vector)
            R_tensor.append(R_matrix)
        R_tensor_obj = Tensor_3D(R_tensor)
        self.R = R_tensor_obj
        return R_tensor_obj
    def forward(self):
        curr_tensor = self.forward_Fs()
        curr_tensor = self.forward_Bs(curr_tensor)
        curr_tensor = self.forward_R(curr_tensor)
        # self.a1 = curr_tensor
        return curr_tensor # this is going to be a1
    def drv_a0(self):
        print("entering drv_a0")
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        nl = self.drv_Bs()[0] # next grad tensor
        assert type(nl) == Tensor_3D
        filt_sidelength = (self.Fs[0].height)
        stride = self.stride
        Fs = self.Fs
        assert type(Fs) == list
        assert type(Fs[0]) == Tensor_3D

        paddilated_nl_tensor_array = []

        for matrix in nl.md:
            dil_matrix = matrix.dilate(stride - 1)
            paddil_matrix = dil_matrix.pad(filt_sidelength - 1)
            paddilated_nl_tensor_array.append(paddil_matrix)
        paddilated_nl_tensor = Tensor_3D(paddilated_nl_tensor_array)
        print("done paddilated")

        flipped_weight_tensors = []

        weight_depth = Fs[0].depth
        for i in range(weight_depth):
            flipped_weight_tensor = []
            for j in range(len(Fs)):
                weight_matrix = Fs[j].md[i]
                assert type(weight_matrix) == Matrix
                weight_matrix_flipped = weight_matrix.rotate().rotate()
                assert type(weight_matrix.vr[0].headc[0]) in [int, float]
                flipped_weight_tensor.append(weight_matrix_flipped)
            flipped_weight_tensors.append(Tensor_3D(flipped_weight_tensor))

        assert type(paddilated_nl_tensor) == Tensor_3D
        assert type(flipped_weight_tensors) == list
        assert type(flipped_weight_tensors[0]) == Tensor_3D
        assert type(flipped_weight_tensors[0].md[0].vr[0].headc[0]) in [int, float]
        try:
            assert len(flipped_weight_tensors) == self.a0.depth
            assert (flipped_weight_tensors[0].depth) == self.a1.depth
        except:
            print("dimensions not switched")
            print("filts have amount {0} and depths {1}".format(len(flipped_weight_tensors),
                                                                flipped_weight_tensors[0].depth))
        print("done flipping weights")
        drv_a0_tensor = self.forward_Fs(a0 = paddilated_nl_tensor,
                                        weight_tensors = flipped_weight_tensors,
                                        stride = 1,
                                        pad = 0)
        try:
            assert drv_a0_tensor.height == self.a0.height
            assert drv_a0_tensor.width == self.a0.width
            assert drv_a0_tensor.depth == self.a0.depth
        except:
            print("filt length {0}".format(str(filt_sidelength)))
            print("stride {0}".format(str(stride)))
            print("filt size {0}".format(str(flipped_weight_tensors[0].height)))
            print("orig size {0}".format(str(paddilated_nl_tensor.height)))
            print("nl orig size {0}".format(str(nl.height)))
            print("drva0 dimens:")
            print(drv_a0_tensor.depth, drv_a0_tensor.height, drv_a0_tensor.width)
            print("self dimens:")
            print(self.a0.depth, self.a0.height, self.a0.width)
            raise AssertionError("UNEQUAL")
        print("done with drv_a0")
        return drv_a0_tensor
    def drv_Fs(self, stride = None, chrt = None): # chain rule of filters
        # chrt is the next tensor in the chain rule
        if stride == None:
            stride = self.stride
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
        if chrt == None:
            chrt = self.drv_Bs()[0] # chain rule tensor,
        drv_tensors = [] # tensors list
        for drv_matrix in chrt.md:
            drv_tensor = []
            for orig_matrix in self.a0.md:
                dilated_drv_matrix = drv_matrix.dilate(stride - 1)
                # dilation for stride
                filt_matrix = self._filt_matrix(orig_matrix, drv_matrix)
                drv_tensor.append(filt_matrix)
            drv_tensors.append(Tensor_3D(drv_tensor))
        #print(drv_tensors)
        return drv_tensors    
    def drv_Bs(self): # returns singleton list of tensors
        return [self.drv_R()]
    def drv_R(self, nl = None): # chain rule of RELU
        if nl == None:
            nl = self.next_layer
        drv_relu_tensor = []
        for matrix in self.R.md:
            drv_matrix = []
            for vector in matrix.vr:
                drv_vector = []
                for x in vector.headc:
                    if x > 0:
                        drv_vector.append(1)
                    else:
                        drv_vector.append(0)
                drv_matrix.append(drv_vector)
            drv_relu_tensor.append(drv_matrix)
        drv_tensor = Tensor_3D(drv_relu_tensor)
        return self.chain_rule_tensor(drv_tensor)
    def backpropagation(self):
        weights_grad = self.drv_Fs()
        biases_grad = self.drv_Bs()
        layer_gradients = [weights_grad, biases_grad]
        return layer_gradients

    def post_creation_organize(self, next_W=None, next_b=None, next_activation=None):
        # organizes everything after when the object is first created
        # OR When the layer is being updated
        # network and layer_rank are the only attributes that must be stated when the new layer is created
        # but we must assume that a0 is given
        # WARNING: forward() is ran in this function

        assert self.a1 != None and self.next_layer != None
        nl = self.next_layer
        new_a1 = self.forward()  # with new a0 and reconfigured W and b
        self.a1.pointer_preserved_change(new_a1)
        nl.a0.pointer_preserved_change(new_a1)


# In[1]:





class Pool_Layer (Cnn_Layer):
    def __init__(self, network = None, layer_rank = 0, a0 = None, 
                 stride = 1, pad = 0):
        super(Pool_Layer, self).__init__(network = network,
                                         layer_rank = layer_rank,
                                         a0 = a0)
        self.a1 = None
        self.max_coords = None # tensor-like 3d list with same dimens as a1, 
        # where entries are coord list of ints - the max coords of max value in a0
        # these coords are only within its respective sub-matrix of self.max_coord, not globally
        self.stride = stride
        self.pad = pad
        self.P = None
        self.prev_drv_shape = None 
    
    def _submatrix_local_max(self, submatrix):
        # input is matrix
        # returns a list of numbers: the max value, followed by the x and y coords
        max_value = None # numeric int or float initially set to None
        max_coords = [-1, -1] # list of 2 positive ints
        # print(submatrix)
        for y in range(len(submatrix.vr)):
            row = submatrix[y]
            for x in range(len(submatrix.vr[0])):
                curr_value = row[x]
                if max_value == None: # if is first value
                    max_value = curr_value
                    max_coords = [x, y]
                else: # if not first (max_value not None), then we can test if max
                    if curr_value <= max_value: # lesser or equal to
                        pass
                    else: # if greater, dethrone the older max_value
                        max_value = curr_value
                        max_coords = [x, y]
        return [max_value, max_coords]
    def forward_P(self):
        # gets the pooled tensor, and the LOCAL max coords
        max_coord_tensor = [] # 4D list (3D tensor with list entries) DO NOT MAKE TENSOR
        pooled_tensor = []
        # print(self.a0) # succ
        for matrix in self.a0:
            temp_pool_matrix = []
            temp_max_coord_matrix = []
            floored_height = len(matrix.vr) - (len(matrix.vr) % self.stride)
            for y1 in range(0, floored_height, self.stride): # height
                temp_pool_row = []
                temp_max_coord_row = []
                floored_width = len(matrix.vr[0]) - (len(matrix.vr[0]) % self.stride)
                for x1 in range(0, floored_width, self.stride): # 
                    x2 = x1 + self.stride
                    y2 = y1 + self.stride
                    submatrix =  matrix.sector(x1, x2 - 1, y1, y2 - 1)
                    # print(submatrix)
                    max_value, max_coord = self._submatrix_local_max(submatrix)
                    temp_pool_row.append(max_value)
                    temp_max_coord_row.append(max_coord)
                    # print("x1: {0} to x2: {1}".format(x1, x2))
                    # print("y1: {0} to y2: {1}".format(y1, y2))
                    # print("")
                temp_pool_matrix.append(temp_pool_row)
                temp_max_coord_matrix.append(temp_max_coord_row)
            max_coord_tensor.append(temp_max_coord_matrix)
            pooled_tensor.append(temp_pool_matrix)
        
        self.max_coords = max_coord_tensor
        pooled_tensor_obj = Tensor_3D(pooled_tensor)
        # print(pooled_tensor_obj) # success
        self.P = pooled_tensor_obj
        return pooled_tensor_obj
    def forward(self):
        curr_tensor = self.forward_P();
        self.a1 = curr_tensor
        return curr_tensor
    def drv_a0(self): #drv_a0 IS the pool gradient
        assert self.prev_drv_shape != None

        if self.next_layer != None:
            nl_drv = self.next_layer.drv_a0()
        else:
            nl_drv = Tensor_3D([[[1 for _ in range(self.a1.width)]
                                 for _ in range(self.a1.height)]
                                for _ in range(self.a1.depth)])
        prev_drv_depth = len(self.prev_drv_shape.md)
        prev_drv_height = len(self.prev_drv_shape.md[0].vr)
        prev_drv_width = len(self.prev_drv_shape.md[0].vr[0].headc)
        
        drv_pool_tensor = [[[0 for _ in range(prev_drv_width)]
                            for _ in range(prev_drv_height)]
                            for _ in range(prev_drv_depth)]
        for z in range(len(self.max_coords)):
            for y in range(len(self.max_coords[0])):
                for x in range(len(self.max_coords[0][0])):
                    temp_local_max_coords = self.max_coords[z][y][x]
                    lmx, lmy = temp_local_max_coords #lm = local max, gm = global max
                    gmx = (x * self.stride) + lmx # global max x
                    gmy = (y * self.stride) + lmy
                    drv_pool_tensor[z][gmy][gmx] = 1 * nl_drv[z][y][x]
        drv_tensor = Tensor_3D(drv_pool_tensor)
        
        return drv_tensor #self.chain_rule_tensor(drv_tensor)
    def post_creation_organize(self, next_W=None, next_b=None, next_activation=None):
        # organizes everything after when the object is first created
        # OR When the layer is being updated
        # network and layer_rank are the only attributes that must be stated when the new layer is created
        # but we must assume that a0 is given
        # WARNING: forward() is ran in this function

        assert self.a1 != None and self.next_layer != None
        nl = self.next_layer
        new_a1 = self.forward()  # with new a0 and reconfigured W and b
        self.a1.pointer_preserved_change(new_a1)
        nl.a0.pointer_preserved_change(new_a1)


# In[15]:


class Fully_Connected_Layer (Cnn_Layer):
    def __init__(self, network = None, layer_rank = 0, a0 = None, 
                 categ_nn = None):
        super(Fully_Connected_Layer, self).__init__(network = network,
                                         layer_rank = layer_rank,
                                         a0 = a0)
        self.categ_nn = categ_nn
        self.a1 = None
    def forward(self, categ_nn = None):
        if categ_nn == None:
            categ_nn = self.categ_nn
        assert categ_nn != None
        FC = (Vector(self.a0.unwrap())).to_matrix()
        self.a1 = FC
        categ_nn.all_layers[0].a0 = FC
        categ_nn.GD4_update_reconfigurations()
        return FC
    def drv_a0(self):
        a0_h = self.a0.height
        a0_w = self.a0.width
        a0_d = self.a0.depth
        
        drv_FC_tensor = []
        categ_a0_drv = self.categ_nn.all_layers[0].drv_a0().unwrap()
        # list of numbers
        for i in range(a0_d):
            drv_FC_matrix = []
            m_area = (a0_h * a0_w)
            m1 = i * m_area
            m2 = m1 + m_area
            matrix_list = categ_a0_drv[m1:m2]
            for j in range(a0_h):
                v1 = j * a0_w
                v2 = v1 + a0_w
                drv_FC_vector = matrix_list[v1 : v2]
                drv_FC_matrix.append(drv_FC_vector)
            drv_FC_tensor.append(drv_FC_matrix)
        return Tensor_3D(drv_FC_tensor)
    def post_creation_organize(self, next_W=None, next_b=None, next_activation=None):
        # organizes everything after when the object is first created
        # OR When the layer is being updated
        # network and layer_rank are the only attributes that must be stated when the new layer is created
        # but we must assume that a0 is given
        # WARNING: forward() is ran in this function


        assert self.a1 != None
        new_a1 = self.forward()  # with new a0 and reconfigured W and b
        self.a1.pointer_preserved_change(new_a1)
        self.categ_nn.all_layers[0].a0.pointer_preserved_change(new_a1)
        self.categ_nn.GD4_update_reconfigurations()
        # categ_nn update_configureations already execudted inside of self.forward()


# In[ ]:


# In[ ]:



class Conv_Categ_NN_Executor:
    def __init__(self, conv_nn_obj, encoder_holder):
        self.conv_nn_obj = conv_nn_obj
        self.encoder_holder = encoder_holder
        self.set_test_verifier_holder(None)
        self.dataset = []
        self.train_dataset = []
        self.test_dataset = []

    def set_test_verifier_holder(self, test_verifier_holder):
        self.test_verifier_holder = test_verifier_holder

    def generate_dataset(self):
        folderpath = "C:/Users/Benson/Desktop/BootlegTensorFlowFolder/convnet_images"
        observations = []
        for filename in os.listdir(folderpath):
            if filename.split(".")[-1] in ["jpeg", "jpg"]:
                image = Image_File(folderpath, filename)
                observation = Observation(image, image.target)
                observations.append(observation)
        repeated_observations = observations # list(np.random.choice(observations, 250))
        self.dataset = (repeated_observations)
        return (repeated_observations)

    def split_dataset(self, train_proportion=None):
        if train_proportion == None:
            train_proportion = 0.8
        random.shuffle(self.dataset)

        dataset_len = len(self.dataset)
        split_index = int(dataset_len * 0.8)
        self.train_dataset = (self.dataset)[:split_index]
        self.test_dataset = (self.dataset)[split_index:]

    def _visualize_filts(self, filts):
        for i in range(len(filts)):
            tensor = filts[i]
            filt_reshaper = []
            for j in range(tensor.height):
                temp_impaled_vectors = []
                
                for depth_layer_matrix in tensor.md:
                    temp_vector = depth_layer_matrix.vr[j]
                    temp_vector_boxed = np.array([[(127.5*x) + 127.5] for x in temp_vector.headc]).astype("float")
                    temp_impaled_vectors.append(temp_vector_boxed)
                final_impaled_vector = np.hstack(tuple(temp_impaled_vectors))
                filt_reshaper.append(final_impaled_vector)
            filt_reshaper_array = np.array(filt_reshaper).astype("uint8")
            # print(filt_reshaper_array)
            filt_image = Image.fromarray(filt_reshaper_array)
            filt_image.save("axototl_filt_image ({0}).jpg".format(i))
        
                
                    
            

    def batch_gradient_descent(self):
        descended_parameters, descnded_categ_parameters = (self.conv_nn_obj).run_batches(
            train_datalist=self.train_dataset,
            num_batches=100)# FOR CATEG
        filts = descended_parameters[0]
        self._visualize_filts(filts)
        return descended_parameters, descnded_categ_parameters # FOR CATEG

    def verify_test_dataset(self, final_parameters, final_categ_parameters):
        assert self.test_verifier_holder != None
        categ_nn_obj = self.conv_nn_obj.all_layers[-1].categ_nn # FOR CATEG
        self.conv_nn_obj.GD3_reconfigure_parameters(final_parameters)
        categ_nn_obj.GD3_reconfigure_parameters(final_categ_parameters) # FOR CATEG
        successes = 0

        for i in range(len(self.test_dataset)):
            trial = self.test_dataset[i]
            assert self.train_dataset.count(trial) == 0
            trial_input_ = trial.info
            trial_target = trial.target
            self.conv_nn_obj.reconfigure_observation(trial_input_, trial_target)
            # print("old input a0 {0}".format(str(self.categ_nn_obj.all_layers[0].a0)))
            # print("old loss a0 {0}".format(str(self.categ_nn_obj.final_loss.a0)))
            # print("old target a0 {0}".format(str(self.categ_nn_obj.final_loss.freq_target)))
            self.conv_nn_obj.GD4_update_reconfigurations()
            loss_stage = self.conv_nn_obj.all_layers[-1].categ_nn.final_loss
            # print("new input a0 {0}".format(str(self.categ_nn_obj.all_layers[0].a0)))
            # print("new loss a0 {0}".format(str(self.categ_nn_obj.final_loss.a0)))
            # print("new target a0 {0}".format(str(self.categ_nn_obj.final_loss.freq_target)))

            verifier = self.test_verifier_holder.apply(loss_stage)
            if verifier == True:
                successes += 1

        print(str(successes) + "/" + str(len(self.test_dataset)))

        
        accuracy = (successes / (len(self.test_dataset)))
        return accuracy

    def setup_dataset(self):
        self.generate_dataset()
        self.split_dataset()

    def main_executor(self):
        self.setup_dataset()
        final_parameters, final_categ_parameters = self.batch_gradient_descent()
        accuracy = self.verify_test_dataset(final_parameters, final_categ_parameters)
        return accuracy

    def MNIST_executor(self, mnist):
        self.setup_MNIST(mnist)
        final_parameters, final_categ_parameters = self.batch_gradient_descent()
        accuracy = self.verify_test_dataset(final_parameters, final_categ_parameters)
        return accuracy

    def setup_MNIST(self, mnist):
        # substitute to both generate_dataset and split_dataset
        def onehot_encode(trainy_i):
            # input is int from 0 to 9
            onehot_list = [0 for _ in range(10)]
            onehot_list[trainy_i] = 1
            return Vector(onehot_list).to_matrix()

        def setup_mnist_image(trainx_i):
            # input is np array
            img_arr = trainx_i - 127.5
            img_arr = img_arr * (1/127.5)
            return Tensor_3D([img_arr.tolist()])
        train_list = []
        test_list = []
        (trainX, trainy), (testX, testy) = mnist.load_data()
        for i in range(len(trainy)):
            trainx_i = trainX[i]
            trainy_i = trainy[i]
            target_onehot= onehot_encode(trainy_i)
            input_tensor = setup_mnist_image(trainx_i)
            train_observation = Observation(input_tensor, target_onehot)
            train_list.append(train_observation)
        for i in range(len(testy)):
            testx_i = testX[i]
            testy_i = testy[i]
            target_onehot= onehot_encode(testy_i)
            input_tensor = setup_mnist_image(testx_i)
            test_observation = Observation(input_tensor, target_onehot)
            test_list.append(test_observation)
        self.train_dataset = train_list
        self.test_dataset = test_list



# 8/3/2020
# nearly changed drv_a0 for Layer class in matrix_network_7_25_2020
# changed the chain rule between pool tensor and FC
# added the batch/gradient functions and maded them accomodatethe tensor-list format for conv net


# 8/4/2020
# SUCCESS!
# categ parameters now update during the gradient descent too
# adding mnist to the exxecutor
# floored range for pooling
# remove the entry_type conversions for __sum__() in the tensorable classes of matrix_7_25-2020
# currerntly stuck at drv_a0 hadamard

# 8/5/2020
# added the optional stride and pad arguments to _filt_matrix and forward_Fs
# wrote down drv_a0
# fixed error in Matrix rotate in matrixed_network_7_25_2020

# Simplifying complexity with np (fixed forward_Fs specifically)
