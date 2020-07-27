
import numpy as np
import math as mt
from phrothud1 import *
from matrixed_network_7_25_2020 import *
import sys, inspect
def print_classes():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            print(obj)


class Tensor_3D:

    def __init__(self, volume):
        # volume can be 2 things: a 3d grid of objects;
        # or a list of matrices
        assert type(volume) == list
        if type(volume[0]) == list:
            if type(volume[0][0]) == list:
                matrix_depths = []
                self.md = [Matrix([Vector(row) for row in array]) for array in volume]
            elif type(volume[0][0]) == Vector:
                self.md = [Matrix(layer) for layer in volume]
        elif type(volume[0]) == Matrix:
            self.md = volume
        self.depth = len(self.md)
        self.height = len(self.md[0].vr)
        self.width = len(self.md[0].vcl)
        self.entry_type = type(self.md[0].vr[0].headc[0])
    def __add__(self, other):
        if type(other) in [float, int]:
            newArray = [(matrix.__add__(other)) for matrix in self.md]
            return Tensor_3D(newArray)  # """
        if type(other) == Tensor_3D:
            assert self.height == other.height and self.depth == other.depth and self.width == other.width
            newArray = []
            for i in range(self.depth):
                self_temp_matrix_layer = self.md[i]
                other_temp_matrix_layer = other.md[i]
                new_temp_matrix_layer = self_temp_matrix_layer.__add__(other_temp_matrix_layer)
                newArray.append(new_temp_matrix_layer)
            return Tensor_3D(newArray)  # """
    def __sub__(self, other):
        if type(other) in [float, int]:
            newArray = [(matrix.__sub__(other)) for matrix in self.md]
            return Tensor_3D(newArray)  # """
        if type(other) == Tensor_3D:
            assert self.height == other.height and self.depth == other.depth and self.width == other.width
            newArray = []
            for i in range(self.depth):
                self_temp_matrix_layer = self.md[i]
                other_temp_matrix_layer = other.md[i]
                new_temp_matrix_layer = self_temp_matrix_layer.__sub__(other_temp_matrix_layer)
                newArray.append(new_temp_matrix_layer)
            return Tensor_3D(newArray)  # """

    def hadamard(self, other):
        if type(other) == Tensor_3D:
            # assert (len(self.vr) == len(other.vr)) and (len(self.vcl) == len(other.vcl))
            newArray = []
            for i in range(self.depth):
                self_temp_matrix_layer = self.md[i]
                other_temp_matrix_layer = other.md[i]
                new_temp_matrix_layer = self_temp_matrix_layer.hadamard(other_temp_matrix_layer)
                newArray.append(new_temp_matrix_layer)
            return Tensor_3D(newArray)  # """

    def __mul__(self, other):
        if type(other) in [int, float]:
            newArray = [(matrix.__mul__(other)) for matrix in self.md]
            return Tensor_3D(newArray)

        elif type(other) == Tensor_3D:
            emptarray = [[0 for _ in range(len(other.vcl))] for _ in range(len(self.vr))]
            newArray = []
            for i in range(self.depth):
                self_temp_matrix_layer = self.md[i]
                other_temp_matrix_layer = other.md[i]
                new_temp_matrix_layer = self_temp_matrix_layer.__mul__(other_temp_matrix_layer)
                newArray.append(new_temp_matrix_layer)
            return Tensor_3D(newArray)  # """

    def sector(self, x_0, x_1, y_0, y_1, z0 = 0, z1 = 0):
        x_criteria = [(x_0 >= 0), (x_1 < (self.width)), (x_0 <= x_1)]
        y_criteria = [(y_0 >= 0), (y_1 < (self.height)), (y_0 <= y_1)]
        z_criteria = [(y_0 >= 0), (y_1 < (self.depth)), (y_0 <= y_1)]

        # assert (all(x_criteria) and all(y_criteria))
        sector = [matrix.sector(x_0, x_1, y_0, y_1) for matrix in self.md]
        return (Tensor_3D(sector))#.__deepcopy__() # revisit
    def raw_form(self):
        # DON'T REMOVE: needed by repr
        return [[row.headc for row in matrix.vr] for matrix in self.md]
    def convert_entry_type(self, new_entry_type):
        if self.entry_type in [int, float, bool, str]:

            return Tensor_3D(
                [[[(new_entry_type)(x) for x in row.headc]
                  for row in matrix.vr]
                 for matrix in self.md]
            )
        else:
            return Tensor_3D(
                [[[(new_entry_type)(x.value) for x in row.headc]
                  for row in matrix.vr]
                 for matrix in self.md]
            )
    def unwrap(self):
        return [x for matrix in self.md for row in matrix.vr for x in row]
    def make_empty_tensor(height, width, depth, entry_type):
        newArray = [[[(entry_type)(0) for _ in range(width)]
                     for _ in range(height)]
                    for _ in range(depth)]
        return Tensor_3D(newArray)
    def __sum__(self):
        entry_type = type(self.md[0].vr[0].headc[0])
        total = (entry_type)(0)
        for matrix in self.md:
            matrix_sum = matrix.__sum__()
            total = total.__add__(matrix_sum)
        return total
    def __copy__(self):
        return Tensor_3D(self.md)
    def __deepcopy__(self):
        return Tensor_3D([matrix.__deepcopy__() for matrix in self.md])
    def __repr__(self):
        # NEEDS NUMPY
        return str(np.array(self.raw_form()))
    def __sum__(self):
        entry_type = self.entry_type
        total = (entry_type)(0)
        for matrix in self.md:
            matrix_sum = matrix.__sum__()
            total = total.__add__(matrix_sum)
        return total

    def get_volume(self):
        # get volume vs __sum__
        # __sum__ adds up all of the items in the tensor.
        # while get_volume counts the amount of items in the tensor
        # get_volume returns an int x where x >= 0
        volume = 0
        for matrix in self.md:
            volume += matrix.get_area()
        return volume
    def equalto(self, unwrapped_ints):
        uw = self.unwrap();
        uwn = [x.value for x in uw]
        return uwn == unwrapped_ints
    def pointer_preserved_change(self, newTensor):
        assert type(self.md[0]) == Matrix # or any class with set_value
        assert type(newTensor.md[0]) == Matrix  # or any class with set_value
        for i in range(len(self.md)):
            self.md[i].pointer_preserved_change(newTensor.md[i])
class Tensor_Derivable:
    def __init__(self):
        pass
    def drv_tensor(self, tensor):
        return None
    def chain_rule_tensor(self, tensor):
        return None

class Tensor_Stage (Tensor_Derivable):
    def __init__(self, orig_tensor):
        self.set_orig_tensor(orig_tensor)
        self.set_next_tensor_stage(None)
        self.set_prev_tensor_stage(None)
        self.set_layer(None) # of int type
        self.set_network(None) # of Convnet type
    def drv_tensor(self, tensor):
        return None
    def chain_rule_tensor(self, tensor):
        return None
        # self.set_layer_rank(layer_rank) # of int type
        # self.set_network(network) # of Convnet type
    def transitioning_chain_rule(self):
        # only activated if the next tensor stage is the fully connected layer
        assert self.next_tensor_stage != None
        assert type(self.next_tensor_stage) == Fully_Connected_Vector
        old_height = self.orig_tensor.height
        old_width = self.orig_tensor.width
        old_depth = self.orig_tensor.depth
        tensor_stack = []
        matrix_stack = []
        # REVISIT
        return
    def dead_end_chain_rule(self,tensor=None):
        # revisit; tester function
        if tensor == None:
            tensor = self.output_tensor
        tensor_h = tensor.height
        tensor_w = tensor.width
        tensor_d = tensor.depth
        tensor_e = tensor.entry_type
        one_cube = [[(tensor_e)(1)
                         for _ in range(tensor_w)]
                        for _ in range(tensor_h)]
        default_tensor = Tensor_3D([one_cube for _ in range(tensor_d)])
        tensor_chain_ruling = default_tensor
        return tensor_chain_ruling
    def set_output_tensor(self, output_tensor):
        self.output_tensor = output_tensor;
    def set_orig_tensor(self, orig_tensor):
        self.orig_tensor = orig_tensor;
    def set_next_tensor_stage(self, next_tensor_stage):
        self.next_tensor_stage = next_tensor_stage
    def set_prev_tensor_stage(self, prev_tensor_stage):
        self.prev_tensor_stage = prev_tensor_stage
    def set_network(self, network):
        self.network = network
    def set_layer(self, layer):
        self.layer = layer
    def update_stage(self, tensor = None):
        # to be overriden
        pass
    def UPDATE(self, tensor = None):
        # the main purpose of UPDATE is to be an enclosing function for update_stage,
        # where we will recursively update the stages by following pointers to the next stage,
        # until we reach none

        # to be overriden
        self.update_stage()
        if self.next_tensor_stage != None:
            self.next_tensor_stage.UPDATE()
class Tensor_Neurons (Tensor_Stage):
    def __init__(self, orig_tensor):
        super(Tensor_Neurons, self).__init__(orig_tensor)
        self.set_output_tensor(orig_tensor)
    def drv_tensor(self, tensor = None):
        # tensor is different from next_tensor_stage.
        # to be correct, it is actually the NEXT NEXT stage's backpropogated matrix
        assert self.next_tensor_stage != None
        # see ctrl-f "have_tensor_neurons_precede_poolbox"

        # more specifically, verify that the next stage is a filt mapping,
        # and not a pool mapping
        unneeded = """
        if False and type(self.next_tensor_stage) != FiltTensor_Mapping:
            if type(self.next_tensor_stage) == PoolTensor_Mapping:
                # if so, we will skip the tensor_neuron's derivatives entirely,
                # since in a real example, there wouldn't be a "tensor_neurons" stage
                # before a pool mapping to begin with.
                return self.next_tensor_stage.chain_rule_tensor() """
        next_tensor_stage = self.next_tensor_stage
        next_filts_count = len(next_tensor_stage.tensor_mappings)

        freq_conv_layer_matrix_obj = Conv_Layer_Matrix(None, None)

        entry_type = self.orig_tensor.entry_type
        zero_matrix = Matrix([[(entry_type)(0)
                               for _ in range(self.orig_tensor.width)]
                              for _ in range(self.orig_tensor.height)])
        temp_summed_drv_matrix = zero_matrix
        dummy_sum_filtbox_counter = 0
        next_drv_matrix_stack = []

        asserter_num_matrix_count = len(next_tensor_stage.tensor_mappings) * next_tensor_stage.tensor_mappings[0].depth
        assert len(next_tensor_stage.mappingboxes) == asserter_num_matrix_count
        for i in range(len(next_tensor_stage.mappingboxes)):
            temp_filtbox = next_tensor_stage.mappingboxes[i]
            temp_tensor_matrix = tensor.md[i // next_filts_count]
            temp_filtbox_mapping_matrix = temp_filtbox.mapping_matrix
            drv_filt_matrix = freq_conv_layer_matrix_obj.drv_matrix(temp_tensor_matrix,
                                                                    temp_filtbox_mapping_matrix)
            temp_summed_drv_matrix = temp_summed_drv_matrix.__add__(drv_filt_matrix)
            dummy_sum_filtbox_counter += 1;
            if dummy_sum_filtbox_counter == next_filts_count:
                next_drv_matrix_stack.append(temp_summed_drv_matrix)
                temp_summed_drv_matrix = zero_matrix
                dummy_sum_filtbox_counter = 0
        return Tensor_3D(next_drv_matrix_stack)

    def chain_rule_tensor(self, tensor = None):
        print("tensor neuron chain rule")
        if self.next_tensor_stage == None:
            return self.dead_end_chain_rule(self.output_tensor);
        if type(self.next_tensor_stage) == FiltTensor_Mapping:
            return self.drv_tensor();
        elif type(self.next_tensor_stage) == PoolTensor_Mapping:
            # if so, we will skip the tensor_neuron's derivatives entirely,
            # since in a real example, there wouldn't be a "tensor_neurons" stage
            # before a pool mapping to begin with.
            return self.next_tensor_stage.chain_rule_tensor()
        elif type(self.next_tensor_stage) == Fully_Connected_Vector:
            # if so, we will skip the tensor_neuron's derivatives entirely,
            # since in a real example, there wouldn't be a "tensor_neurons" stage
            # before a pool mapping to begin with.
            return self.next_tensor_stage.chain_rule_tensor()
    def update_stage(self, tensor = None):
        if tensor == None:
            # we are assuming that the self.orig_tensor was indriectly
            # affected by the previous stage's output tensor being reconfigured
            tensor = self.orig_tensor
        self.set_output_tensor(tensor)
class Tensor_Mapping (Tensor_Stage, Tensor_Derivable):
    def __init__(self, orig_tensor):
        super(Tensor_Mapping, self).__init__(orig_tensor)
        self.set_output_tensor(None)
        self.set_mappingboxes([]) # DO NOT TOUCH THIS BEFORE USING APPLY_TENSOR_MAPPINGS
    def apply_tensor_mappings(self):
        # to be overriden
        return None;
    def set_output_tensor(self, output_tensor):
        self.output_tensor = output_tensor;
    def set_mappingboxes(self, mappingboxes):
        # this should not be initialized at the start yet
        self.mappingboxes = mappingboxes
    def add_mappingbox(self, mappingbox):
        self.mappingboxes.append(mappingbox)
    def finalize(self):
        self.set_output_tensor(self.apply_tensor_mappings())
    def update_stage(self, tensor = None):
        if tensor == None:
            tensor = self.orig_tensor # hopefully updated indirectly from prev
        old_mappingbox_length = len(self.mappingboxes)
        # when we run self.finalize(), it will add new mappingboxes to self.mappingbox,
        # so we prevent that by recutting self.mappingbox to it's old length afterwards
        for x in self.mappingboxes:
            x.finalize()
        new_output_tensor = self.apply_tensor_mappings()
        self.output_tensor.pointer_preserved_change(new_output_tensor)
        # when you run the apply_tensor_mappings(), the self.mappingboxes will double in size
        # the new mappingboxxs were created because we ran the function add_mappingbox
        # every time we made a new mappingbox in the function apply_tensor_mappings()
        # so you must reconfigure the tensor attributes of the older items of self.mappingboxes
        # with the new tensor attributes (while preserving attributes)

        # revisit
        # edit: ok, so after some assert test runs, we don't need to actually rechange each mappingbox
        # pointer-preservedly. (second method under update_mappingboxes_str)
        # simply making the new mappingboxes as the old ones works too: (first method)
        self.mappingboxes = self.mappingboxes[old_mappingbox_length:]
        update_mappingboxes_str = """
        old_mappingboxes = self.mappingboxes[0: old_mappingbox_length]
        new_mappingboxes = self.mappingboxes[old_mappingbox_length:];

        for i in range(len(self.mappingboxes[0: old_mappingbox_length])):

            (self.mappingboxes[i].orig).pointer_preserved_change(new_mappingboxes[i].orig)
            (self.mappingboxes[i].mapping).pointer_preserved_change(new_mappingboxes[i].mapping)
            (self.mappingboxes[i].output).pointer_preserved_change(new_mappingboxes[i].output)
        self.mappingboxes = self.mappingboxes[0: old_mappingbox_length]
        # """

class FiltTensor_Mapping (Tensor_Mapping, Tensor_Derivable):
    # revisit: MIGHT BE DONE WRONG
    def __init__(self, orig_tensor):
        super(FiltTensor_Mapping, self).__init__(orig_tensor)
        self.set_tensor_mappings([]) # must be manually initialized after object creation
        self.set_strides(None) # must be manually initialized after object creation
        self.set_pad(0)
    def set_tensor_mappings(self, tensor_mappings):
        self.tensor_mappings = tensor_mappings
    def get_tensor_mappings(self):
        return self.tensor_mappings
    def add_tensor_mapping(self, tensor_mapping):
        assert type(tensor_mapping) == Tensor_3D
        self.tensor_mappings.append(tensor_mapping)
        # if tensor mapping was for filtboxes, you'd append a lot
        # if it was for relu or poolbox, you'd append only once
        # or several for each layer if you're going the filtbox-centric way
    def set_pad(self, pad):
        self.pad = pad
    def set_strides(self, strides = None):
        if strides == None:
            self.strides = [1 for _ in range(len(self.tensor_mappings))]
        else:
            self.strides = strides
    def add_strides(self, stride):
        assert type(self.strides) == list
        self.stride.append(stride)
    def apply_tensor_mappings(self):
        final_summed_filtcubes_stack = []
        # order might matter, check if so
        for i in range(len(self.tensor_mappings)):
            weight_tensor = self.tensor_mappings[i]
            temp_stride = self.strides[i]
            # revisit: former args were
            # (weight_tensor, self.orig_tensor.__deepcopy__(), temp_stride)
            # but we removed .__deepcopy__()
            summed_filtcube = self.apply_weight_tensor(weight_tensor,
                                                       self.orig_tensor,
                                                       temp_stride)
            final_summed_filtcubes_stack.append(summed_filtcube)
        return Tensor_3D(final_summed_filtcubes_stack)
    def apply_weight_tensor(self, weight_tensor, orig_tensor, stride):
        # orig_tensor = self.orig_tensor.__deepcopy__();
        assert orig_tensor.depth == weight_tensor.depth
        entry_type = orig_tensor.entry_type
        feature_map_height = (((orig_tensor.height - weight_tensor.height) + (2*self.pad)) // stride) + 1
        feature_map_width = (((orig_tensor.width - weight_tensor.width) + (2*self.pad)) // stride) + 1

        zero_matrix = [[(entry_type)(0)
                        for _ in range(feature_map_width)]
                       for _ in range(feature_map_height)]
        summed_filtboxes_matrix = Matrix(zero_matrix)
        for i in range(orig_tensor.depth):
            temp_orig_matrix = orig_tensor.md[i]
            temp_weight_matrix = weight_tensor.md[i]

            temp_filtbox = FiltBox(temp_orig_matrix, temp_weight_matrix, stride=1, bias=0)
            # will be eventually reduced to a 2d tensor to stack on final mapping tensor
            temp_filtbox.finalize()
            self.add_mappingbox(temp_filtbox)
            summed_filtboxes_matrix = summed_filtboxes_matrix.__add__(temp_filtbox.output)
        return summed_filtboxes_matrix
    def drv_tensor(self, tensor):
        # tensor will be NEXT LAYER'S drv_matrix
        assert self.next_tensor_stage != None
        next_grad_tensor = self.next_tensor_stage.chain_rule_tensor() # is a tensor
        base_filts_count = len(self.mappingboxes) // len(self.tensor_mappings)
        # aka, depth of input_tensor
        base_strides_count = base_filts_count
        drv_tensors_stack = []
        drv_matrix_stack = []
        for i in range(len(self.mappingboxes)):
            temp_filtbox = self.mappingboxes[i]
            temp_filtbox_orig_matrix = temp_filtbox.orig
            # revisit
            # assert filtbox.orig in self.orig_tensor and in self.prev_tensor_stage.output_tensor
            temp_next_matrix = next_grad_tensor.md[ i//base_filts_count]
            temp_filter_stride = self.strides[i // base_strides_count]
            # temp_filter_stride += 1 # just to test
            temp_expanded_next_matrix = temp_next_matrix.dilate(d = temp_filter_stride - 1)

            drv_matrix_filtbox = FiltBox(temp_filtbox_orig_matrix,
                                         temp_expanded_next_matrix,
                                         stride=1, bias=0)
            drv_matrix_filtbox.finalize()
            drv_matrix_stack.append(drv_matrix_filtbox.output)
            if len(drv_matrix_stack) == base_filts_count:
                drv_tensors_stack.append(Tensor_3D(drv_matrix_stack))
                drv_matrix_stack = []
        return (drv_tensors_stack) # NOT A Tensor
    def chain_rule_tensor(self, tensor=None ,next_tensor_mappings=None):
        print("Filt chain rule")
        return self.drv_tensor(self.next_tensor_stage.chain_rule_tensor())
class PoolTensor_Mapping (Tensor_Mapping, Tensor_Derivable):
    def __init__(self, orig_tensor):
        super(PoolTensor_Mapping, self).__init__(orig_tensor)
        self.set_pool_strides([])
        self.set_pool_dims([])
    def set_pool_strides(self, pool_strides):
        self.pool_strides = pool_strides
    def set_pool_dims(self, pool_dims):
        self.pool_dims = pool_dims
        # dimension of new pooled matrix
    def apply_tensor_mappings(self):
        if (self.pool_strides == []) or (self.pool_dims == []):
            raise AssertionError("Set the pool_strides and pool_dims!")
        orig_tensor = self.orig_tensor
        pool_strides = self.pool_strides
        pool_dims = self.pool_dims
        assert len(self.pool_strides) == len(self.pool_dims)
        assert orig_tensor.depth == len(self.pool_strides)
        final_poolcubes_stack = [] # layer order might matter, so check on that - revisit

        for i in range(orig_tensor.depth):
            temp_orig_matrix = orig_tensor.md[i]#.__deepcopy__()
            temp_pool_dim = pool_dims[i]
            temp_pool_stride = pool_strides[i]
            temp_poolbox = PoolBox(temp_orig_matrix,
                                   Matrix.IdentityMatrix(temp_pool_dim),
                                   temp_pool_stride)
            temp_poolbox.finalize()
            self.add_mappingbox(temp_poolbox)
            final_poolcubes_stack.append(temp_poolbox.output)
        return Tensor_3D(final_poolcubes_stack)
    def drv_tensor(self ,tensor):
        # tensor will be output_tensor
        print("assuming the poolboxes were not changed beforehand")
        drv_matrix_stack = []
        for i in range(tensor.depth):
            temp_poolbox = self.mappingboxes[i]
            # revisit
            # we might need to use the previous (left_adjacent tensor)
            # in the chain rule as the orig_tensor, not the temp_poolbox orig
            # if there is more than one poolbox in a convolution
            # then doing this below will only reshape the matrixx to the
            # locally previous tensorstage, not the overall derivative
            temp_drv_matrix = temp_poolbox.drv_matrix(temp_poolbox.orig)
            drv_matrix_stack.append(temp_drv_matrix)
        return Tensor_3D(drv_matrix_stack)
    def chain_rule_tensor(self, tensor=None ,next_tensor_mappings=None):
        # PERFORM AFTER SETTING SELF.OUTPUT AND SELF.NEXT_MAPPING
        print("pool chain rule")
        if tensor == None:
            tensor = self.next_tensor_stage.chain_rule_tensor()
        drv_curr_tensor = (self.drv_tensor(tensor))
        return drv_curr_tensor # revisit
class ReluTensor_Mapping (Tensor_Mapping, Tensor_Derivable):
    def __init__(self, orig_tensor):
        super(ReluTensor_Mapping, self).__init__(orig_tensor)
    def apply_tensor_mappings(self):
        orig_tensor = self.orig_tensor
        final_relucubes_stack = [] # layer order might matter, so check on that - revisit
        # temp_tensor_mapping = self.tensor_mappings[0]
        for i in range(orig_tensor.depth):
            temp_orig_matrix = orig_tensor.md[i]#.__deepcopy__()
            temp_relubox = ReluBox(temp_orig_matrix)
            temp_relubox.finalize()
            self.add_mappingbox(temp_relubox)
            final_relucubes_stack.append(temp_relubox.output)
        return Tensor_3D(final_relucubes_stack)
    def drv_tensor(self ,tensor):
        # tensor will be output_tensor
        drv_matrix_stack = []
        for i in range(tensor.depth):
            temp_relubox = self.mappingboxes[i]
            temp_drv_matrix = temp_relubox.drv_matrix(temp_relubox.output)
            drv_matrix_stack.append(temp_relubox.output)
        return Tensor_3D(drv_matrix_stack)
    def chain_rule_tensor(self, tensor=None ,next_tensor_mappings=None):
        # PERFORM AFTER SETTING SELF.OUTPUT AND SELF.NEXT_MAPPING
        print("relu chain rule")
        next_tensor_stage = self.next_tensor_stage
        tensor = self.output_tensor
        tensor_chain_rulings = self.next_tensor_stage.chain_rule_tensor()
        drv_curr_tensor = (self.drv_tensor(tensor))
        return drv_curr_tensor.hadamard(tensor_chain_rulings)
class BiasTensor_Mapping (Tensor_Mapping, Tensor_Derivable):
    def __init__(self, orig_tensor):
        super(BiasTensor_Mapping, self).__init__(orig_tensor)
        self.set_tensor_mappings(None)
        self.set_mappingboxes([]) # DO NOT TOUCH THIS BEFORE USING APPLY_TENSOR_MAPPINGS
    def set_tensor_mappings(self, tensor_mappings):
        self.tensor_mappings = tensor_mappings
    def get_tensor_mappings(self):
        return self.tensor_mappings
    def add_tensor_mapping(self, tensor_mapping):
        assert type(tensor_mapping) == Tensor_3D
        self.tensor_mappings.append(tensor_mapping)
    def apply_tensor_mappings(self):
        assert self.tensor_mappings != None
        orig_tensor = self.orig_tensor
        bias_tensor = self.tensor_mappings[0] # layer order might matter, so check on that - revisit
        # temp_tensor_mapping = self.tensor_mappings[0]
        final_bias_translated_matrix_stack = []
        for i in range(orig_tensor.depth):
            temp_orig_matrix = orig_tensor.md[i]#.__deepcopy__()

            temp_bias_matrix = bias_tensor.md[i]#.__deepcopy__()
            temp_biasbox = BiasBox(temp_orig_matrix, temp_bias_matrix)
            temp_biasbox.finalize()
            self.add_mappingbox(temp_biasbox)
            final_bias_translated_matrix_stack.append(temp_biasbox.output)
        return Tensor_3D(final_bias_translated_matrix_stack)
    def drv_tensor(self ,tensor):
        # tensor will be output_tensor
        drv_matrix_stack = []
        for i in range(tensor.depth):
            temp_relubox = self.mappingboxes[i]
            temp_drv_matrix = temp_relubox.drv_matrix(temp_relubox.output)
            drv_matrix_stack.append(temp_relubox.output)
        return Tensor_3D(drv_matrix_stack)
    def chain_rule_tensor(self, tensor=None ,next_tensor_mappings=None):
        print("bias chain rule")

        # PERFORM AFTER SETTING SELF.OUTPUT AND SELF.NEXT_MAPPING
        return self.next_tensor_stage.chain_rule_tensor()
class Fully_Connected_Vector (Tensor_Stage):
    def __init__(self,orig_tensor = None):
        super(Fully_Connected_Vector, self).__init__(orig_tensor=orig_tensor)
        self.set_temp_target_matrix(None)
    def set_temp_target_matrix(self, temp_target_matrix):
        self.temp_target_matrix = temp_target_matrix
    def set_output_tensor(self, output_tensor):
        self.output_tensor = output_tensor;

    def set_output_neurons(self, output_neurons = None):
        # THIS IS WHERE CONNECTION WITH THE CATEG_NN 
        # is column matrix vector
        assert self.temp_target_matrix != None
        assert self.network != None
        if output_neurons == None:
            # self.output_tensor.unwrap() = column vector of pixels
            output_neurons = Matrix([[x.value] for x in self.output_tensor.unwrap()])
        
        assert type(output_neurons) == Matrix
        assert len(output_neurons.vr[0].headc) == 1
        self.output_neurons = output_neurons;

        print(type(self.output_neurons.vr[0].headc[0]))

        categ = self.network.categ_nn
        categ.reconfigure_observation(self.output_neurons, self.temp_target_matrix)
        categ.GD4_update_reconfigurations()
        
    def apply_tensor_mappings(self):
        # flattens 3d tensor from previous stage to a 1d vector
        column_matrix = []
        for matrix in self.orig_tensor.md:
            for vector in matrix.vr:
                for pixel in vector.headc:
                    column_matrix.append([pixel])
        flattened_tensor = Tensor_3D([column_matrix])
        return flattened_tensor
    def finalize(self):
        flattened_tensor = self.apply_tensor_mappings()
        self.set_output_tensor(flattened_tensor)
        self.set_output_neurons()

    def update_stage(self, tensor = None):
        # we are assuming that orig_tensor is already updated
        # so finalize reupdates the output_tensor
        if tensor == None:
            tensor = self.orig_tensor # assuming self.orig_tensor was updated during precedeing part of update
        self.finalize()

    def chain_rule_tensor(self, tensor = None):
        print("Fully Connected chain rule")
        if tensor == None:
            tensor = self.prev_tensor_stage.output_tensor
            # tensor neuron. but we can use self.orig_tensor too
        old_height = tensor.height
        old_width = tensor.width
        old_depth = tensor.depth
        unwrapped = self.output_tensor.unwrap()
        unwrapped_neurons = self.network.categ_nn.all_layers[0].a0.to_vector()
        unwrapped_len = len(unwrapped) # or unwrapped.dimens
        temp_tensor = []

        print(self.network.categ_nn.all_layers[0].a0)
        base_layer_drv = self.network.categ_nn.all_layers[0].drv_a0().to_vector()
        print("base_layer_drv")
        print(base_layer_drv)
        for i in range(0, unwrapped_len, unwrapped_len // old_depth):
            temp_matrix = []
            temp_matrix_dimens = old_height * old_width
            for j in range(i, i + temp_matrix_dimens, temp_matrix_dimens// old_height):
                temp_vector_chain_rule = base_layer_drv.headc[j: j + old_width]
                print("temp_vector chain rule")
                print(temp_vector_chain_rule)
                temp_matrix.append(temp_vector_chain_rule)
            temp_tensor.append(temp_matrix)
        return Tensor_3D(temp_tensor)

class Tensor_Layer:
    def __init__(self, layer_rank = 0, network = None, orig_tensor = None):
        self.set_layer_rank(layer_rank) # of int type
        self.set_network(network) # of Convnet type
        self.set_orig_tensor(orig_tensor) # Tensor_3D type
        self.set_tensor_stages([])
    def set_network(self, network):
        if network != None:
            network.layers.append(self)
        self.network = network
    def set_layer_rank(self, layer_rank):
        print("layer rank {0}".format(layer_rank))
        assert type(layer_rank) == int and layer_rank >= 0
        self.layer_rank = layer_rank
    def set_tensor_stages(self, tensor_stages):
        # use this afterwards
        self.tensor_stages = tensor_stages
    def add_tensor_stage(self, tensor_stage):
        self.tensor_stages.append(tensor_stage)
    def set_orig_tensor(self, tensor):
        self.orig_tensor = tensor
    def set_final_tensor(self, final_tensor):
        self.final_tensor = final_tensor
    def run_layer(self, input_tensor = None, tensor_neuron_stage = None):
        # outputs the final tensor
        # but this will be overriden
        return None
    def starting_layer(self, input_tensor):
        # if there is no tensor_neuron_stage
        stage_0 = Tensor_Neurons(input_tensor)
        return stage_0
    def update_layer(self):
        # after doing:
        # sets to weight_tensors_lists or bias_tensor_list
        # WE WILL NOT DO THE UPDATES TO the causal attributes during the updating itself

        for i in range(len(self.tensor_stages)):
            self.tensor_stages[i].update_stage()
class Convolving_Tensor_Layer (Tensor_Layer):
    # filtboxes always signify a new layer is already happening
    # thus the previous stage, the base_tensor is the start of it
    # consist of list of tensors
    # filtbox might be a problem in terms of sequence, but no problems besides that

    def __init__(self, layer_rank = 0, network=None,orig_tensor = None,
                 weight_tensors_list = None, bias_tensor_list = None, strides_list = None):
        super(Convolving_Tensor_Layer, self).__init__(layer_rank = layer_rank,
                                                      network = network,
                                                      orig_tensor = orig_tensor)
        self.set_weight_tensors_list(weight_tensors_list) # list of tensors
        self.set_bias_tensor_list(bias_tensor_list) # tensor of real numbers
        self.set_strides_list(strides_list)
    def set_weight_tensors_list(self, weight_tensors_list):
        # self.network.weight_tensors_vectors.append(weight_tensors_list)
        self.weight_tensors_list = weight_tensors_list
    def add_weight_tensor(self, weight_tensor):
        self.weight_tensors.append(weight_tensor)
    def set_strides_list(self, strides_list):
        assert len(strides_list) == len(self.weight_tensors_list)
        self.strides_list = strides_list
    def add_stride(self, stride):
        self.strides_list.append(stride)
    def set_bias_tensor_list(self, bias_tensor_list):
        # list of tensors. We are calling it a "list" even though it has only one item
        # just to give it polymorphic conformity to the weights_tensor_list
        if bias_tensor_list == None:
            self.bias_tensor_list = None;
            return None;

        # assert (bias_tensor_list[0]).depth == len(self.weight_tensors_list)
        self.bias_tensor_list = bias_tensor_list

    def run_layer(self, input_tensor=None, tensor_neuron_stage = None):
        # assert self.network != None
        assert self.weight_tensors_list != None
        assert self.bias_tensor_list != None
        assert self.strides_list != None

        if input_tensor == None:
            input_tensor = self.orig_tensor
        if tensor_neuron_stage == None:
            tensor_neuron_stage = self.starting_layer(input_tensor)
        # part 0: Tensor_Neurons set_up and pointer set up
        stage_0 = tensor_neuron_stage
        stage_0.set_network(self.network)
        self.add_tensor_stage(stage_0)
        stage_0.set_layer(self);

        # part 1: filtbox
        input_tensor = stage_0.output_tensor

        filttensor_mapper = FiltTensor_Mapping(input_tensor)
        filttensor_mapper.set_tensor_mappings(self.weight_tensors_list)
        filttensor_mapper.set_strides(self.strides_list) # runs default if
        filttensor_mapper.finalize()

        # part1z: pointer updates:
        filttensor_mapper.set_network(self.network)
        self.add_tensor_stage(filttensor_mapper)
        filttensor_mapper.set_layer(self);
        stage_0.set_next_tensor_stage(filttensor_mapper)
        filttensor_mapper.set_prev_tensor_stage(stage_0)

        input_tensor = filttensor_mapper.output_tensor

        # part 1.5: biasbox:
        biastensor_mapper = BiasTensor_Mapping(input_tensor)
        biastensor_mapper.set_tensor_mappings(self.bias_tensor_list)
        biastensor_mapper.finalize()

        # part 1.5: pointer updates:
        biastensor_mapper.set_network(self.network)
        self.add_tensor_stage(biastensor_mapper)
        biastensor_mapper.set_layer(self);
        filttensor_mapper.set_next_tensor_stage(biastensor_mapper)
        biastensor_mapper.set_prev_tensor_stage(filttensor_mapper)

        input_tensor = biastensor_mapper.output_tensor

        # part 2: relubox:
        relutensor_mapper = ReluTensor_Mapping(input_tensor)
        relutensor_mapper.finalize()

        relutensor_mapper.set_network(self.network)
        self.add_tensor_stage(relutensor_mapper)
        relutensor_mapper.set_layer(self);

        biastensor_mapper.set_next_tensor_stage(relutensor_mapper)
        relutensor_mapper.set_prev_tensor_stage(biastensor_mapper)

        # part 2z:


        final_tensor = relutensor_mapper.output_tensor
        # see CTRL-F "have_tensor_neurons_precede_poolbox"
        # we might return a tensor_stage object,
        # specifically the one to start off the next layer,
        rebridged_mapper = Tensor_Neurons(final_tensor)
        rebridged_mapper.set_network(self.network)
        rebridged_mapper.set_layer(None) # DO NOT PUT SELF; WE WILL SET_LAYER IN NEXT LAYER
        rebridged_mapper.set_next_tensor_stage(None)
        rebridged_mapper.set_prev_tensor_stage(relutensor_mapper)
        relutensor_mapper.set_next_tensor_stage(rebridged_mapper)
        return rebridged_mapper
    def update_layer1(self, new_tensor):
        # only do this after doing a set_value for weight_tensors_list or bias

        # if we had done it in the where the input tensor of each stage has the same id as the
        # output tensor of the stage preceding it, we need not carry it over,
        # we just need to update the output tensor of one stage, and then run the next one normally

        pass
class Pooling_Tensor_Layer (Tensor_Layer):
    # we are treating this as a separate layer just in case it fails spectacularly
    # but it must still be preceded by any ConvNet_Tensor_Conv_Layer
    # has no weights or bias, but we will still backpropogate thru it
    def __init__(self, layer_rank = 0, network=None, orig_tensor = None,
                 pool_dim_list = [], stride_list = []):
        super(Pooling_Tensor_Layer, self).__init__(layer_rank = layer_rank,
                                                   network = network,
                                                   orig_tensor = orig_tensor)
        self.set_pool_dim_list(pool_dim_list)
        self.set_stride_list(stride_list)
    def set_pool_dim_list(self, pool_dim_list):
        self.pool_dim_list = pool_dim_list
    def set_stride_list(self, stride_list):
        self.stride_list = stride_list
    def run_layer(self ,input_tensor = None, tensor_neuron_stage = None):
        assert (self.pool_dim_list != []) and (self.stride_list != [])
        pool_dim_list = self.pool_dim_list
        stride_list = self.stride_list
        assert len(stride_list) == len(pool_dim_list)
        # BEYOND THIS POINT, YOU SHALL NOT CHANGE POOL_DIM_LIST OR STRIDE_LIST
        if input_tensor == None:
            input_tensor = self.orig_tensor
        # for the pool mapping
        assert len(stride_list) == input_tensor.depth
        if input_tensor == None:
            input_tensor = self.orig_tensor
        if tensor_neuron_stage == None:
            tensor_neuron_stage = self.starting_layer(input_tensor)

        stage_0 = tensor_neuron_stage
        stage_0.set_network(self.network)
        self.add_tensor_stage(stage_0)
        stage_0.set_layer(self);

        input_tensor = stage_0.output_tensor

        pooltensor_mapper = PoolTensor_Mapping(input_tensor)
        pooltensor_mapper.set_network(self.network)
        pooltensor_mapper.set_layer(self)
        pooltensor_mapper.set_pool_dims(pool_dim_list)
        pooltensor_mapper.set_pool_strides(stride_list)
        self.add_tensor_stage(pooltensor_mapper)

        stage_0.set_next_tensor_stage(pooltensor_mapper)
        pooltensor_mapper.set_prev_tensor_stage(stage_0)

        result_tensor = pooltensor_mapper.apply_tensor_mappings()
        pooltensor_mapper.set_output_tensor(result_tensor)

        final_tensor = pooltensor_mapper.output_tensor

        rebridged_mapper = Tensor_Neurons(final_tensor)
        rebridged_mapper.set_network(self.network)
        rebridged_mapper.set_layer(None)  # DO NOT PUT SELF; WE WILL SET_LAYER IN NEXT LAYER
        rebridged_mapper.set_next_tensor_stage(None)
        rebridged_mapper.set_prev_tensor_stage(pooltensor_mapper)
        pooltensor_mapper.set_next_tensor_stage(rebridged_mapper)
        return rebridged_mapper
class Fully_Connected_Vector_Layer (Tensor_Layer):
    def __init__(self, layer_rank = 0, network=None, orig_tensor = None, target_matrix = None):
        super(Fully_Connected_Vector_Layer, self).__init__(layer_rank, network, orig_tensor)
        # the minute we initialize this is when we start doing the
        # things like, setup_first_layer, push_layer, finalize_cost_layer

        # num_categs = self.network.dataset.num_categories
        self.set_target_matrix(target_matrix) # = [0 for _ in range(num_categs)]
        # categ.push_layer(numbases = num_categs, weight = 1)
        # categ.finalize_cost_layer(target = default_target_vector)
    def set_target_matrix(self, target_matrix):
        self.target_matrix = target_matrix
    def run_layer(self, input_tensor = None, tensor_neuron_stage = None):
        # we will need to use:
        # softmax independently
        # configure_target independently
        # configure_first_bases independently
        # configure_bases independently
        if input_tensor == None:
            input_tensor = self.orig_tensor
        if tensor_neuron_stage == None:
            tensor_neuron_stage = self.starting_layer(input_tensor)
        stage_0 = tensor_neuron_stage
        stage_0.set_network(self.network)
        self.add_tensor_stage(stage_0)
        stage_0.set_layer(self);

        input_tensor = stage_0.output_tensor

        fully_connected_stage = Fully_Connected_Vector(input_tensor)
        print(fully_connected_stage)
        fully_connected_stage.set_network(self.network)
        fully_connected_stage.set_layer(self)
        # categ_nn must already have been created, down to the last loss layer
        fully_connected_stage.set_temp_target_matrix(self.target_matrix)
        fully_connected_stage.finalize()
        self.add_tensor_stage(fully_connected_stage)

        stage_0.set_next_tensor_stage(fully_connected_stage)
        fully_connected_stage.set_prev_tensor_stage(stage_0)
        
        return fully_connected_stage

class ConvNet:
    def __init__(self, dataset = None, categ_nn = None):
        self.set_layers([])
        self.set_conv_parameters_tensors_list([]) # all weight_tensors_list's and bias_tensor_list's
        # i.e. ArrayList<ArrayList<Tensor_Mapping>> where Tensor_Mapping is mainly
        # FiltTensor_Mapping -> BiasTensorMapping, in that alternating order with Filt first
        self.set_dataset(dataset)
        self.set_categ_nn(categ_nn)
    def set_layers(self, layers):
        self.layers = layers
    def set_dataset(self, dataset):
        self.dataset = dataset
    def set_categ_nn(self, categ_nn):
        self.categ_nn = categ_nn
    def set_conv_parameters_tensors_list(self, conv_parameters_tensors_list):
        # initially empty
        self.conv_parameters_tensors_list = conv_parameters_tensors_list

    def create_conv_gradient(self): # CLEANed
        # creates the gradients by backpropogation
        gradient_tensors_list = []
        # will be a list of lists of tensors
        for i in range(len(self.layers)):
            temp_layer = self.layers[i]
            # in this list, the bias tensor is the last tensors
            # all of the other tensors are the weight_tensors for filt
            if type(temp_layer) == Convolving_Tensor_Layer:
                temp_layer_parameters = [] # list of tensors
                filt_tensor_stage = temp_layer.tensor_stages[1]  # filt_stage = second stage, or 1
                bias_tensor_stage = temp_layer.tensor_stages[2]  # filt_stage = thirdt stage, or 2
                filt_tensor_chain_ruled = filt_tensor_stage.chain_rule_tensor() # list of tensors
                bias_tensor_chain_ruled = bias_tensor_stage.chain_rule_tensor() # tensor, so put this in list
                temp_layer_parameters.extend(filt_tensor_chain_ruled)
                temp_layer_parameters.append(bias_tensor_chain_ruled)
                gradient_tensors_list.append(temp_layer_parameters)
        return gradient_tensors_list
    def finalize_conv_gradient_tensors_list(self, gradient_tensors_list): # cleaned
        # subtracts the current parameter tensor list by the gradient_tensor_list made by create_conv_gradient
        # list of lists of drv tensors, where each sublist has all of the tensors of the filter, plus the bias tensor at the end
        
        updated_parameters_tensors_list = [] # related to gradient_tensors_list
        # list of lists of drv tensors
        conv_layers = [layer for layer in self.layers if type(layer) == Convolving_Tensor_Layer]
        for i in range(len(gradient_tensors_list)):
            temp_updated_layer_parameters = [] # is the recreated equivalent of temp_layer_parameters
            temp_layer = conv_layers[i]

            temp_parameter_tensors_list = gradient_tensors_list[i]
            temp_weight_tensors_list = temp_parameter_tensors_list[:-1]
            # the bias_tensor_list is always the last tensor
            temp_bias_tensor_list = temp_parameter_tensors_list[-1:]

            filt_tensor_stage = temp_layer.tensor_stages[1] # filt_stage = second stage, or 1
            
            bias_tensor_stage = temp_layer.tensor_stages[2] # filt_stage = thirdt stage, or 2
            
            for j in range(len(filt_tensor_stage.get_tensor_mappings())):
                temp_filt_tensor = filt_tensor_stage.get_tensor_mappings()[j]
                temp_weight_tensor = temp_weight_tensors_list[j]
                temp_weight_grad = temp_filt_tensor.__sub__(temp_weight_tensor)
                # temp_filt_tensor.pointer_preserved_change(temp_grad_tensor)
                temp_updated_layer_parameters.append(temp_weight_grad)

            temp_bias_old = (bias_tensor_stage.get_tensor_mappings()[0])
            temp_bias_tensor = (temp_bias_tensor_list[0])
            temp_bias_grad = temp_bias_old.__sub__(temp_bias_tensor)
            temp_updated_layer_parameters.append(temp_bias_grad)
            
            updated_parameters_tensors_list.append(temp_updated_layer)
        return updated_parameters_tensors_list
    
    def configure_conv_parameters_tensors_list(self, updated_parameters_tensors_list): # cleaned
        # reconfigures 
        conv_layers = [layer for layer in self.layers if type(layer) == Convolving_Tensor_Layer]
        x = 0
        for i in range(len(updated_parameters_tensors_list)):
            temp_layer = conv_layers[i]
            if type(temp_layer) == Convolving_Tensor_Layer:
                temp_parameter_tensors_list = updated_parameters_tensors_list[x]
                temp_weight_tensors_list = temp_parameter_tensors_list[:-1]
                # the bias_tensor_list is always the last tensor
                temp_bias_tensor_list = temp_parameter_tensors_list[-1:]

                filt_tensor_stage = temp_layer.tensor_stages[1] # filt_stage = second stage, or 1
                bias_tensor_stage = temp_layer.tensor_stages[2] # filt_stage = thirdt stage, or 2
                for j in range(filt_tensor_stage.get_tensor_mappings()):
                    temp_filt_tensor = filt_tensor_stage.get_tensor_mappings()[j]
                    temp_weight_tensor = temp_weight_tensors_list[j]
                    temp_filt_tensor.pointer_preserved_change(temp_weight_tensor)
                bias_tensor_stage.get_tensor_mappings()[0].pointer_preserved_change(temp_bias_tensor_list[0])
                x += 1
            else:
                pass
        stage_0 = self.layers[0].tensor_stages[0]
        stage_0.UPDATE() # recursive updating

    def create_categ_gradient(self): # cleaned
        # returns the gradient vector for the categ weights and biases part
        categ_gradient = self.categ_nn.GD1_create_gradient()
        return categ_gradient # list of matrices
    def configure_categ_parameters(self, categ_gradient): # cleaned
        # combines gradient_conv_parameters_tensors_list
        # and configure_conv_parameters_tensors_list in one function
        # but for categ parameters
        updated_categ_parameters = self.categ_nn.GD2_finalise_gradient(categ_gradient)
        # self.categ_nn.configure_all_parameters(updated_categ_parameters) # redundant
        self.categ_nn.GD3_reconfigure_parameters(updated_categ_parameters)
        self.categ_nn.GD4_update_reconfigurations()
    
    def CGD1_get_all_gradient_parameters(self):
        # gets gradients for both conv part and categ part
        conv_gradient = self.create_conv_gradient()
        categ_gradient = self.create_categ_gradient()
        print("conv_gradient is \n"+str(conv_gradient))
        print(categ_gradient)
        return conv_gradient, categ_gradient
    def CGD2_update_all_gradient_parameters(self, conv_gradient, categ_gradient):
        # updates gradient parameters for both conv part and categ part
        print((conv_gradient))
        print(categ_gradient)
        
        new_conv_parameters = self.gradient_conv_parameters_tensors_list(conv_gradient)
        self.configure_conv_parameters_tensors_list(new_conv_parameters)
        self.configure_categ_parameters(categ_gradient)
        return self.conv_parameters_tensors_list, self.categ_nn.parameters







def import_test():
    Ghost = Tensor_3D([Matrix.IdentityMatrix(2), Matrix.IdentityMatrix(2)])
    Ghoststr = """[[[1 0]
  [0 1]]

 [[1 0]
  [0 1]]]"""
    assert str(Ghost) == Ghoststr
    return True
cube = [
    [[1, 0, 0],
         [0, 1, 0],
         [0, 1, 0]],
        [[1, 1, 1],
         [0, 1, 0],
         [0, 1, 0]],
        [[0, 1, 1],
         [1, 0, 0],
         [1, 1, 0]]
]
tc = Tensor_3D(cube)
ptc = tc.convert_entry_type(Pixel)
filtcube = [
    [[1, 0],
     [0, 1]],
    [[1, 1],
     [0, 1]],
    [[0, 1],
     [0, 1]]
]
fc = Tensor_3D(filtcube)
pfc = fc.convert_entry_type(Pixel)
filtcube2 = [
    [[1, 0],
     [0, 1]],
    [[1, 1],
     [0, 1]],
    [[0, 1],
     [0, 1]]
]
fc2 = Tensor_3D(filtcube2)
pfc2 = fc2.convert_entry_type(Pixel)
def init_test():
    cube = [[[1, 0, 0],
             [0, 1, 0],
             [0, 1, 0]],
            [[1, 1, 1],
             [0, 1, 0],
             [0, 1, 0]],
            [[0, 1, 1],
             [1, 0, 0],
             [1, 1, 0]]]
    tc = Tensor_3D(cube)
    ptc = tc.convert_entry_type(Pixel)

    strptc = """[[[P1 P0 P0]
  [P0 P1 P0]
  [P0 P1 P0]]

 [[P1 P1 P1]
  [P0 P1 P0]
  [P0 P1 P0]]

 [[P0 P1 P1]
  [P1 P0 P0]
  [P1 P1 P0]]]"""
    assert str(ptc) == strptc
    print("Init_test successful")
    return True
def tensor_test():
    assert import_test()
    assert init_test()
def filttensor_test():
    ftm = FiltTensor_Mapping(ptc)
    ftm.add_tensor_mapping(pfc)
    ftm.add_tensor_mapping(pfc2)
    ftm.set_strides()
    ftm.finalize()
    # print(ftm.output_tensor)
    fbmstr = """[[[P2 P0]
 [P1 P1]], [[P3 P2]
 [P2 P1]], [[P1 P1]
 [P1 P0]], [[P2 P0]
 [P1 P1]], [[P3 P2]
 [P2 P1]], [[P1 P1]
 [P1 P0]]]"""
    assert str([(fb.output) for fb in ftm.mappingboxes]) == fbmstr

    assert str(ftm.output_tensor) == """[[[P6 P3]
  [P4 P2]]

 [[P6 P3]
  [P4 P2]]]"""
    return True
# donut delet
poolstr = """
[[[P1 P1]
  [P1 P1]]

 [[P1 P1]
  [P1 P1]]

 [[P1 P1]
  [P1 P1]]]"""
def pooltensor_test(poolstr):
    pcube = [
        [[0, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 1]],
        [[1, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 1]],
        [[0, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 1]]
    ]
    tc = Tensor_3D(pcube)
    ptc = tc.convert_entry_type(Pixel)
    ptm = PoolTensor_Mapping(ptc)
    ptm.set_pool_strides([2, 2, 2])
    ptm.set_pool_dims([2, 2, 2])
    ptm.finalize()
    assert "\n" +str(ptm.output_tensor) == poolstr
    pmb = """[[[P1 P1]
 [P1 P1]], [[P1 P1]
 [P1 P1]], [[P1 P1]
 [P1 P1]]]"""
    assert pmb == str([pb.output for pb in ptm.mappingboxes])
    return True
def relutensor_test():
    rcube = [
        [[-1, 2],
         [0, 3]],
        [[5, -6],
         [7, -8]]
    ]
    tc = Tensor_3D(rcube)
    rtc = tc.convert_entry_type(Pixel)
    rtm = ReluTensor_Mapping(rtc)
    rtm.finalize()
    # print(rtm.output_tensor)
    relustr= """
[[[P0 P2]
  [P0 P3]]

 [[P5 P0]
  [P7 P0]]]"""
    assert "\n" +str(rtm.output_tensor) == relustr
    rmb = """[[[P0 P2]
 [P0 P3]], [[P5 P0]
 [P7 P0]]]"""
    assert str([rb.output for rb in rtm.mappingboxes]) == rmb

    return True
def mapping_tests():
    global poolstr
    filttensor_test()
    pooltensor_test(poolstr)
    relutensor_test()
    print("flood tensor_mappable resevoirs")
mapping_tests()
def make_bias_matrix(height, width, number):
    array = [[number for i in range(width)]
            for i in range(height)]
    return Matrix(array)
def conv_layer_stages_test(ptc, pfcl):
    cvntw = ConvNet()
    numfilts = 2
    sl = [1]
    pfcs = [pfc]
    switcher = 1
    if switcher == 1:
        numfilts = 2;
        sl = [1, 1]
        pfcs = pfcl #[pfc, pfc2]
    elif switcher == 0:
        numfilts = 1;
        sl = [1]
        pfcs = pfcl[0] #[pfc]
    bias_matrix_stack = []
    predicted_filt_dim = 2 # expected dimension of filtbox

    orig_dim1 = ptc.height
    map_dim1 = pfcl[0].height
    out_dim1 = ((orig_dim1 - map_dim1)//sl[0]) + 1

    orig_dim2 = ptc.width
    map_dim2 = pfcl[0].width
    out_dim2 = ((orig_dim2 - map_dim2)//sl[0]) + 1

    for i in range(1, numfilts+1):
        temp_bias_matrix = make_bias_matrix(out_dim1, out_dim2, i)
        bias_matrix_stack.append(temp_bias_matrix)
    bias_tensor = Tensor_3D(bias_matrix_stack)
    btl = [bias_tensor]
    ntw = None
    ctl1 = Convolving_Tensor_Layer(orig_tensor = ptc, layer_rank = 1,
                                   network=cvntw, weight_tensors_list = pfcs,
                                   bias_tensor_list = btl,
                                   strides_list = sl)
    finmapper = ctl1.run_layer(ptc)
    print(" please remember to set_prev_tensor_stage and set_layer on output stage object")
    print(" in the next layer")
    fintensor = finmapper.orig_tensor
    # print(fintensor)
    print("layeral reservoirs flooded")
    return (ctl1, finmapper)
ctl1, ft = conv_layer_stages_test(ptc, [pfc, pfc2])
ftmo1 = ctl1.tensor_stages[1]
assert ft.orig_tensor.equalto([7, 4, 5, 3, 8, 5, 6, 4])
five_cube = [
    [[1, 0, 0, 1, 1],
     [0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0],
     [1, 1, 0, 0, 1],
     [0, 0, 1, 1, 0]],
    [[1, 1, 1, 0, 1],
     [0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0],
     [1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1]],
    [[0, 1, 1, 0, 1],
     [1, 0, 0, 1, 0],
     [1, 1, 0, 0, 1],
     [0, 1, 0, 1, 0],
     [0, 0, 1, 0, 0]]
]
five_tc = Tensor_3D(five_cube)
ptc_fiver = five_tc.convert_entry_type(Pixel)
ctl1_fiver, ft_fiver = conv_layer_stages_test(ptc_fiver, [pfc, pfc2])
try:
    assert ft_fiver.output_tensor.equalto(
        [7, 4, 4, 5,
         5, 3, 5, 3,
         5, 4, 3, 6,
         4, 6, 4, 3,
         8, 5, 5, 6,
         6, 4, 6, 4,
         6, 5, 4, 7,
         5, 7, 5, 4]
    )
except:
    print("Error2")
    print(ft_fiver.output_tensor)
def pooling_layer_stages_test(prev_stage, pool_dim_list, stride_list):
    cvntw = prev_stage.network
    ptl1 = Pooling_Tensor_Layer(orig_tensor = prev_stage.output_tensor,
                                layer_rank = 1,
                                network=cvntw,
                                pool_dim_list = [], stride_list = [])
    ptl1.set_pool_dim_list(pool_dim_list)
    ptl1.set_stride_list(stride_list)
    finmapper = ptl1.run_layer(tensor_neuron_stage = prev_stage)
    try:
        assert finmapper.output_tensor.equalto([7, 5, 6, 6, 8, 6, 7, 7])
    except:
        print("Error6")
        print(prev_stage.output_tensor)
        print(finmapper.output_tensor)

    # print(ptl1.tensor_stages[1].mappingboxes[0].aggregation)
    return ptl1, finmapper
pool_stages_layer, pool_stages_final_stage = pooling_layer_stages_test(ft_fiver, [2, 2], [2, 2])
print(" checking relu with negatives")
cubeA = [
    [[-1, 0, 0],
         [0, -1, 0],
         [0, -1, 0]],
        [[1, 1, -1],
         [0, 1, 0],
         [0, -1, 0]],
        [[0, -1, 1],
         [-1, 0, 0],
         [-1, -1, 0]]
]
tcA = Tensor_3D(cubeA)
ptcA = tcA.convert_entry_type(Pixel)
ctlA, ftA= conv_layer_stages_test(ptcA, [pfc, pfc2])
assert ftA.orig_tensor.equalto([1, 2, 0, 1, 2, 3, 0, 2])
ftmoA = ctlA.tensor_stages[1]
filt_chainrule_list = (ftmoA.chain_rule_tensor())
assert filt_chainrule_list[0].equalto([-2, 0, -3, -1,
                                       4, -1, 1, 1,
                                       -2, 1, -2, 0])
def pool_layer_drv_test(layer, stage = None):
    poolbase = layer.tensor_stages[0]
    conv_relustage = poolbase.prev_tensor_stage
    conv_biasstage = conv_relustage.prev_tensor_stage
    conv_filtstage = conv_biasstage.prev_tensor_stage

    pool_poolstage = layer.tensor_stages[-1]
    pooled_pool_ch_r = pool_poolstage.chain_rule_tensor()
    print("pooled stage's drv_matrix")
    try:
        assert (pooled_pool_ch_r).equalto(
        [1, 0, 0, 1,
         0, 0, 0, 0,
         0, 0, 0, 1,
         0, 1, 0, 0,
         1, 0, 0, 1,
         0, 0, 0, 0,
         0, 0, 0, 1,
         0, 1, 0, 0]
        )
    except:
        print("Error")
        print(pooled_pool_ch_r)

    pooled_filt_ch_r = conv_filtstage.chain_rule_tensor()
    pooled_bias_ch_r = conv_biasstage.chain_rule_tensor()

    print("pooled stage's drv_matrix")
    try:
        assert (pooled_bias_ch_r).equalto(
        [7, 0, 0, 5,
         0, 0, 0, 0,
         0, 0, 0, 6,
         0, 6, 0, 0,
         8, 0, 0, 6,
         0, 0, 0, 0,
         0, 0, 0, 7,
         0, 7, 0, 0]
        )
    except:
        print("Error")
        print(pooled_bias_ch_r)

    try:
        assert (pooled_filt_ch_r[0]).equalto(
            [24, 5, 0, 24, 13, 18, 5, 19, 6, 18, 18, 6])
    except:
        print("Error")
        print(pooled_bias_ch_r)
    print("pool tensor derival resevoirs flooded")
    return True
pool_layer_drv_test(layer = pool_stages_layer)
print("tensor-derival resevoirs flooded")
def pointer_test_layer(conv_layer):
    pointers_attr = conv_layer.__dict__
    dpa_orig = pointers_attr['orig_tensor']
    dpa_tensor_stages = pointers_attr['tensor_stages']
    dpa_weights = pointers_attr['weight_tensors_list']
    dpa_bias = pointers_attr['bias_tensor_list']
    tmap0, tmap1, tmap2, tmap3 = dpa_tensor_stages
    # tmap0 = neurons
    # tmap1 = filt
    # tmap2 = bias
    # tmap3 = relu
    # print(id(tmap0.orig_tensor), id(tmap1.orig_tensor), id(dpa_orig))
    try:
        assert len(list(set([id(tmap0.orig_tensor), id(tmap1.orig_tensor), id(dpa_orig)]))) == 1
    except:
        print("ERROR")
        print(id(tmap0.orig_tensor), id(tmap1.orig_tensor), id(dpa_orig))
        return;
    tmap1_mapboxlist = tmap1.mappingboxes
    try:
        assert id(dpa_weights[0]) == id(tmap1.tensor_mappings[0])
        # print(id(dpa_weights[0]), id(tmap1.tensor_mappings[0]))
        for j in range(len(tmap1.tensor_mappings)):
            assert id(dpa_weights[j]) == id(tmap1.tensor_mappings[j])
    except:
        print("ERROR")
        print(id(dpa_weights[0]), id(tmap1.tensor_mappings[0]))
        for j in range(len(tmap1.tensor_mappings)):
            print(id(dpa_weights[j]), id(tmap1.tensor_mappings[j]))

    for i in range(0, len(tmap1.mappingboxes)):
        tmap1_temp_mapfiltbox = tmap1.mappingboxes[i]
        tmap1_temp_mapfiltbox_orig = tmap1_temp_mapfiltbox.orig # tensor
        tmap1_temp_mapfiltbox_map = tmap1_temp_mapfiltbox.mapping # tensor

        origtensor_tempmatrix = tmap1.orig_tensor.md[i%3]
        tensormaps = tmap1.tensor_mappings[i//3]
        tensormaps_tempmatrix = tensormaps.md[i%3] # Matrix

        weighttensor = dpa_weights[i//3]
        weighttensor_tempmatrix = weighttensor.md[i%3] # Matrix

        try:
            assert tmap1_temp_mapfiltbox_orig is origtensor_tempmatrix

            weight_tensors_mapping_ids = [(id(tmap1_temp_mapfiltbox_map)),
             (id(tensormaps_tempmatrix)),
             (id(weighttensor_tempmatrix))]
            assert len(list(set(weight_tensors_mapping_ids))) == 1

        except:
            print("ERROR")

            print("orig_tensors")
            print(tmap1_temp_mapfiltbox_orig)
            print(origtensor_tempmatrix)
            print(" ")

            print("filters")

            print(id(tmap1_temp_mapfiltbox_map))
            print(id(tensormaps_tempmatrix))
            print(id(weighttensor_tempmatrix))

            print(" ")
    print("matches weights and origs")
    try:
        biasmapboxes = tmap2.mappingboxes
        assert tmap2.orig_tensor is tmap1.output_tensor
    except:
        print("ERROR")
        print(tmap2.orig_tensor)
    try:
        Tru = tmap2.orig_tensor.equalto([0, 1, -2, 0, 0, 1, -2, 0])
        assert True
    except:
        print("ERROR  BUT MIGHT BE BECAUSE GIVEN CTL IS NOT == ctl1")
    for i in range(len(biasmapboxes)):
        biasmapbox = biasmapboxes[i]
        try:
            assert biasmapbox.orig is tmap2.orig_tensor.md[i]
        except:
            print("ERROR")
            print(biasmapbox.orig)
            print(" mapbox orig above, dpa orig below")
            print(tmap2.orig_tensor.md[i])
            print(" ")
        try:
            assert biasmapbox.mapping is dpa_bias[0].md[i]
        except:
            print("ERROR")
            print(biasmapbox.mapping)
            print(" mapbox mapping above, dpa bias below")
            print(dpa_bias[0].md[i])
            print(" ")
    try:
        Tru = tmap2.output_tensor.equalto([1, 2,-1, 1, 2, 3, 0, 2])
        assert True
    except:
        print("ERROR  BUT MIGHT BE BECAUSE GIVEN CTL IS NOT == ctl1")
        print(tmap2.output_tensor)
    # RELU
    relumapboxes = tmap3.mappingboxes
    try:
        # assert tmap3.orig_tensor.equalto([0, 1,-2, 0, 0, 1, -2, 0])
        assert tmap3.orig_tensor is tmap2.output_tensor
    except:
        print("ERROR")

        print(tmap3.orig_tensor)
    for i in range(len(relumapboxes)):
        relumapbox = relumapboxes[i]
        try:
            assert relumapbox.orig is tmap3.orig_tensor.md[i]
        except:
            print("ERROR")
            print(relumapbox.orig)
            print(" mapbox orig above, orig below")
            print(tmap3.orig_tensor.md[i])
            print(" ")
        try:
            assert relumapbox.mapping.vr[0].headc[0] == 1
            # universally usable magic reference,
            # since relubox mapping is always [[1]]
        except:
            print("ERROR")
            print(relumapbox.mapping)
    try:
        Tru = tmap3.output_tensor.equalto([1, 2,0, 1, 2, 3, 0, 2])
        assert True
    except:
        print("ERROR  BUT MIGHT BE BECAUSE GIVEN CTL IS NOT == ctl1")
        print(tmap3.output_tensor)
    print("Pointer resevoirs flooded")
filtcubeB = [
    [[6, 0],
     [0, 3]],
    [[1, -8],
     [0, -4]],
    [[0, 1],
     [0, 1]]
]
fc2B = Tensor_3D(filtcubeB)
pfc2B = fc2B.convert_entry_type(Pixel)
def post_updated_pointer_test_pool_layer(layer):
    stage_0 = layer.tensor_stages[0]
    pool_stage = layer.tensor_stages[1]
    assert pool_stage.mappingboxes[0].aggregation.max_entries == [[0,0],[1,3],[3,0],[3,2]]

    stage_0.orig_tensor.md[0].vr[0].headc[0].set_value(0)
    layer.update_layer()
    try:
        assert layer.tensor_stages[-1].output_tensor.equalto([5, 5, 6, 6, 8, 6, 7, 7])
    except:
        print("error")
        print(layer.tensor_stages[-1].output_tensor)
    pool_stage_grad = pool_stage.chain_rule_tensor()
    assert pool_stage.mappingboxes[0].aggregation.max_entries == [[0,1],[1,3],[3,0],[3,2]]
    try:
        assert pool_stage_grad.equalto(
        [0, 0, 0, 1,
         1, 0, 0, 0,
         0, 0, 0, 1,
         0, 1, 0, 0,
         1, 0, 0, 1,
         0, 0, 0, 0,
         0, 0, 0, 1,
         0, 1, 0, 0]
        )
    except:
        print("error")
        print(pool_stage_grad)
    print("Poolbox update pointers resovoirs flooded")
post_updated_pointer_test_pool_layer(layer = pool_stages_layer)
# ctl1.weight_tensors_list[1] = pfc2B
# ^ this cascades the updates to all parts EXCEPT mappingbox



# ctl1.orig_tensor.pointer_preserved_change(ptcA)
# ^ this updates only the first orig_tensor, not tensor_stages or tensor_stage.mappingboxes
# ctl1.orig_tensor.md[0].vr[0].headc[0].set_value(999)
# ^ this cascades the updates to all 3 parts
# ctl1.weight_tensors_list[1].md[0].vr[0].headc[0].set_value(-66)
# ^ this cascades the updates to all parts
ctl1.orig_tensor.pointer_preserved_change(ptcA)
# ^ this cascades the updates to all 3 parts
pointer_test_layer(ctl1)
ctl1.update_layer()
def post_updated_pointer_test():
    st1, st2, st3, st4 = ctl1.tensor_stages
    assert st1.orig_tensor is st1.output_tensor
    assert st2.orig_tensor is st1.output_tensor
    assert st2.output_tensor.equalto([0, 1, -2, 0,0,  1, -2, 0])
    assert st3.orig_tensor.equalto([0,1,-2,0,0,1,-2,0])
    assert st3.output_tensor.equalto([1, 2, -1, 1, 2, 3, 0, 2])
    assert id(st3.output_tensor) == id(st4.orig_tensor)
    assert (st3.output_tensor) is (st4.orig_tensor)
    assert st4.orig_tensor.equalto([1, 2, -1, 1, 2, 3, 0, 2])
    assert st4.output_tensor.equalto([1, 2, 0, 1, 2, 3, 0, 2])

post_updated_pointer_test()
print(" REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE ")
pointer_test_layer(ctl1)


def bridge_test1():
    five_cube = [
        [[1, 0, 0, 1, 1],
         [0, 1, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [1, 1, 0, 0, 1],
         [0, 0, 1, 1, 0]],
        [[1, 1, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 0, 1],
         [1, 0, 1, 0, 1]],
        [[0, 1, 1, 0, 1],
         [1, 0, 0, 1, 0],
         [1, 1, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0]]
    ]
    five_tc = Tensor_3D(five_cube)
    ptc_fiver = five_tc.convert_entry_type(Pixel)
    filtcube = [
        [[1, 0],
         [0, 1]],
        [[1, 1],
         [0, 1]],
        [[0, 1],
         [0, 1]]
    ]
    fc = Tensor_3D(filtcube)
    pfc = fc.convert_entry_type(Pixel)
    filtcube2 = [
        [[1, 0],
         [0, 1]],
        [[1, 1],
         [0, 1]],
        [[0, 1],
         [0, 1]]
    ]
    fc2 = Tensor_3D(filtcube2)
    pfc2 = fc2.convert_entry_type(Pixel)
    ctl1_fiver, ft_fiver = conv_layer_stages_test(ptc_fiver, [pfc, pfc2])
    pool_stages_layer, pool_stages_final_stage = pooling_layer_stages_test(ft_fiver,
                                                                           [2, 2], [2, 2])


# 7/7/2020:
# today's log: made dilated feedforwards
# derivable tensors tested and passed
# pointer tests: all works
# UPDATED pointer test: all works
# except for the mappingboxes for biasbox, and maybe relubox


# 7/9/2020:
# trying to finish poolbox, but has resolved tensor_neuron stage problem
# after dentist: we stopped at line 729; pool layer's run_layer method still outdated, please update


# 7/12/2020:
# connected the convnet with the categ_nn successfully
# image reading complete
# UNFINISHED TASK: implementing softmax


