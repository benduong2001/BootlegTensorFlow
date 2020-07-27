from matrixed_network_7_25_2020 import *
import numpy as np
import math as mt


class Pixel:
    def __init__(self, value):
        self.set_value(value)
    def set_value(self, value):
        self.value = value;
    def set_network(self, network):
        network.neurons.append(self)
        self.network = network
    def set_layer(self, layer):
        layer.neurons.append(self)
        self.layer = layer
    def set_FC_neuron(self, FC_neuron):
        self.FC_neuron = FC_neuron
    def __gt__(self, x):
        if type(x) in [int, float]:
            return self.value > x;
        else:
            return self.value > x.value;
    def __lt__(self, x):
        if type(x) in [int, float]:
            return self.value < x;
        else:
            return self.value < x.value;
    def __le__(self, x):
        if type(x) in [int, float]:
            return self.value <= x;
        else:
            return self.value <= x.value;
    def __ge__(self, x):
        if type(x) in [int, float]:
            return self.value >= x;
        else:
            return self.value >= x.value;

    def __add__(self, x):
        if type(x) in [int, float]:
            return Pixel(self.value + x)
        else:
            return Pixel(self.value + x.value);
    def __mul__(self, x):
        if type(x) in [int, float]:
            return Pixel(self.value * x)
        else:
            return Pixel(self.value * x.value);
    def __copy__(self):
        return Pixel(self.value)
    def __deepcopy__(self):
        return Pixel(self.value)
    
    def __repr__(self):
        return "P{0}".format(str(self.value))
class Derivable_Matrix:
    def __init__(self):
        pass
    def drv_matrix(self): # Revisit
        pass
    def chain_rule_matrix(self):
        pass
    
class Conv_Layer_Matrix (Derivable_Matrix):
    def __init__(self, matrix, mappingbox=None):
        self.set_matrix(matrix)
        self.set_mappingbox(None)
        self.set_prev_layer(None)
        self.set_next_layer(None)
    def set_matrix(self, matrix):
        self.matrix = matrix
    def set_mappingbox(self, mappingbox):
        self.mappingbox = mappingbox
    def set_prev_layer(self, layer):
        self.prev_layer = layer
    def set_next_layer(self, layer):
        self.next_layer = layer
    
    def drv_matrix(self, input_matrix, mappingbox=None):
        # input_matrix is NOT self.matrix
        # mappingbox in this case, is the self.mappingbox
        # in this case, filtbox
        if mappingbox == None:
            mappingbox = self.mappingbox
        assert issubclass(mappingbox, MappingBox)        
        "https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710"
        map_stride = mappingbox.stride
        
        # map_height = len(mappingbox.mapping.vr)
        map_width = len(mappingbox.mapping.vcl)

        mapping_matrix = mappingbox.mapping
        
        flipped_mapping = self.rotate(self.rotate(mappingbox.mapping))
        expanded_matrix = self.dilate(input_matrix, map_stride - 1)
        expanded_matrix = self.pad(expanded_matrix, map_width)
        grad_mappingbox = FiltBox(expanded_matrix, flipped_mapping, map_stride)
        return grad_mappingbox.output
    def chain_rule_matrix(self):
        return;
    
    def dilate(self, matrix, d=0):
        mw = len(matrix.vr)
        assert mw > 1
        entry_type = type(matrix[0].headc[0])
        nmw = mw + (d * (mw - 1))
        nm = [[(entry_type)(0) for _ in range(nmw)] for _ in range(nmw)]
        for i in range(0, nmw, d+1):
            for j in range(0, nmw, d+1):
                nm[i][j] = matrix[i // (d + 1)].headc[j // (d + 1)]
        return Matrix(nm)
    def pad(self, matrix, padn = 0):
        assert padn >= 0
        entry_type = type(matrix[0].headc[0])
        mw = len(matrix.vr)
        nmw = mw + (2 * padn)
        nm = [[(entry_type)(0) for _ in range(nmw)] for _ in range(nmw)]
        for i in range(padn, padn + mw):
            for j in range(padn, padn + mw):
                nm[i][j] = matrix.vr[i - padn].headc[j - padn]
        return Matrix(nm)
    def rotate(self, matrix):
        # rotate clockwise by 90 degrees
        arrayed = [[row.headc] for row in matrix.vr]
        #assert len(array) == len(array[0])
        width = len(arrayed[0])
        sideways_array = [[row[i] for row in arrayed][::-1]
                          for i in range(width)]
        return Matrix(sideways_array)

class MappingBox (Derivable_Matrix):
    # for 2D SQUARE IMAGES ONLY

    # after first initializing, run obj.apply_mapping() immediately!
    def __init__(self, orig_matrix, mapping_matrix, aggregation, stride):
        assert type(orig_matrix) == Matrix;
        assert type(mapping_matrix) == Matrix
        assert issubclass(type(aggregation), Mapping_Aggregator_Function_Holder)
        # aggregation is a function with the input of a list of numbers
        assert type(stride) == int and stride > 0
        self.set_orig(orig_matrix);
        self.set_mapping(mapping_matrix);
        self.set_stride(stride)
        self.set_aggregation(aggregation)

        self.set_output_dim(self.find_output_dim(orig_matrix, mapping_matrix, stride))
        self.set_output_dim2(self.find_output_dim2(orig_matrix, mapping_matrix, stride))

        empty_output_matrix = self.make_empty_output(self.output_dim)
        self.set_output(empty_output_matrix)
        # self.finalize()
        # we could have just put empty_output_matrix for self.output instead - revisit
    def set_orig(self, orig_matrix):
        self.orig = orig_matrix
    def set_mapping(self, mapping_matrix):
        self.mapping = mapping_matrix
    def set_aggregation(self, aggregation):
        self.aggregation = aggregation
    def set_stride(self, stride):
        self.stride = stride
    def make_empty_output(self, height, width = None):
        if width == None:
            width = height
        entry_type = type(self.orig.rows[0][0])
        empty_array = [[(entry_type)(0) for _ in range(width)]
                       for _ in range(height)]
        return Matrix(empty_array)
    def find_output_dim(self, orig_matrix, mapping_matrix, stride):
        orig_dim = len(orig_matrix.vr)
        mapping_dim = len(mapping_matrix.vr)
        output_dim = mt.floor((orig_dim - mapping_dim) / stride) + 1

        return output_dim
    def find_output_dim2(self, orig_matrix, mapping_matrix, stride):
        orig_dim2 = len(orig_matrix.vr[0].headc)
        mapping_dim2 = len(mapping_matrix.vr[0].headc)
        output_dim2 = mt.floor((orig_dim2 - mapping_dim2) / stride) + 1

        return output_dim2
    def set_output_dim(self, output_dim):
        self.output_dim = output_dim
    def set_output_dim2(self, output_dim2):
        self.output_dim2 = output_dim2
    def set_output(self, output_matrix):
        self.output = output_matrix
    def finalize(self):
        # applies mapping and sets output to result
        self.set_output(self.apply_mapping(self.orig, self.mapping, self.output))
    def update(self):
        # assumes that the self.orig or self.mapping has been newly set
        # we will just apply_mapping as if everything's normal, and then do a entry_pointer_preserved change
        new_matrix = self.apply_mapping(self.orig, self.mapping, self.output)

        width = len(self.output.vr[0].headc)
        for i in range(len(self.output.vr)):
            row = self.output.vr[i]
            for j in range(width):
                assert type(self.output.vr[i].headc[j]) == Pixel
                (self.output.vr[i].headc[j]).set_value(new_matrix.vr[i].headc[j].value)
    def apply_mapping(self, orig, mapping, output):
        if orig == None:
            orig = self.orig
        if mapping == None:
            mapping = self.mapping
        if output == None:
            output = self.output
        
        o = len(orig.vr)
        old_f = len(output.vr)
        f = len(mapping.vr)
        # o = len(orig.vr[0].headc)
        # f = len(output.vr[0].headc)
        fn = f - 1;
        old_end_bound1 = o - f + 1
        old_end_bound2 = min(o - ((f % o)) + 1, o) # please REVISIT
        end_bound = old_end_bound2
        stride = self.stride;
        return self.AMhelper_0(orig, mapping, output, end_bound, stride)
    def AMhelper_0(self,orig, mapping, output, end_bound, stride):
        # helper function 1 of apply_mapping (AM)
        # returns output matrix

        # print("    DELEGATED    ")

        if type(self.aggregation) == BiasBox_Helper_Summer:
            output = self.AMhelper0_5_diverging_course_for_biasbox(orig, mapping)
            return output;
        
        for i in range(0, end_bound, stride):
            for j in range(0, end_bound, stride):
                sector_matrix = self.AMhelper1_get_sectored_matrix(i, j, orig, mapping)
                filt_sector = self.AMhelper2_get_filt_matrix(sector_matrix, mapping)
                filt_sector_unwrapped = self.AMhelper3_get_unwrapped_filt_sector(filt_sector)
                entry = self.AMhelper4_get_entry(self.aggregation, filt_sector_unwrapped)
                if type(self.aggregation) == PoolBox_Helper_Summer:
                    self.AMhelper5_poolbox_register_maxcoords(i, j,stride, self.aggregation)
                self.AMhelper6_assign_entry(i, j, stride, output, entry)
        return output
    def AMhelper0_5_diverging_course_for_biasbox(self, orig, mapping):
        # helper function 0.5 of apply_mapping (AM)

        # ONLY UNDER THE CONDITION that aggregation is of BiasBox_Helper_Summer
        # returns matrix translated according to mapping (which is the bias matrix)
        # orig, mapping are matrix of the same shape
        # this is a completely different path from the rest of the mappingboxes,
        # hence it is outside of the main forloop in AMhelper_0

        translated_matrix = orig.__add__(mapping)
        return translated_matrix
    def AMhelper1_get_sectored_matrix(self, i, j, orig, mapping):
        # helper function 1 of apply_mapping (AM)
        # returns sectored matrix
        # i, j are the base coordinate ints
        # orig is the matrix being sectored
        # mapping is the smaller matrix whose dimensions
        # will form the sectored matrix
        sector_matrix = orig.sector(i, i + (len(mapping.vr[0].headc)-1),
                                    j, j + (len(mapping.vr)-1))
        return sector_matrix
    def AMhelper2_get_filt_matrix(self, sector_matrix, mapping):
        # helper function 2 of apply_mapping (AM)
        # returns a matrix - the filtered sector matrix
        # sector matrix is the sectored matrix
        # mapping is the other matrix. both needs the same shape
        # they are then hadamard multiplied into the filtered sector

        # BUT IF THE AGGREGATION IS  biasBOX, then add componentwise
        filt_sector = sector_matrix.hadamard(mapping)
        return filt_sector
    def AMhelper3_get_unwrapped_filt_sector(self, filt_sector):
        # helper function 3 of apply_mapping (AM)
        # returns the list which is the filt_sector unwrapped from 2d to 1d
        return filt_sector.unwrap()
    def AMhelper4_get_entry(self, aggregation, unwrapped_filt_sector):
        # helper function 4 of apply_mapping (AM)
        # returns entry, which is a Pixel classtype
        # aggregation is of type Mapping_Aggregator_Function_Holder
        # unwrapped_filt_sector is a list
        entry = aggregation.apply(unwrapped_filt_sector)
        return entry
    def AMhelper5_poolbox_register_maxcoords(self, i, j, stride, aggregation):
        # helper function 5 of apply_mapping (AM)
        # returns nothing
        # This is a conditional function that only operates when the mapping is for a Poolbox
        # We are cataloguing the location of the max entries for later reasons when we backpropagate
        # print(aggregation.max_indices)
        x_c = i + (aggregation.max_indices[-1] % stride)
        y_c = j + (aggregation.max_indices[-1] // stride)
        aggregation.max_entries.append([x_c, y_c])
        return None;
    def AMhelper6_assign_entry(self, i, j, stride, output, entry):
        # This basically assigns entries to new matrix
        # returns nothing
        # i, j, stride are ints, where i,j is the coordinates
        # output is the matrix
        # entry is the parameter to be assigned
        output.vr[j // stride].headc[i // stride] = entry
        output.rows[j // stride][i // stride] = entry

class FiltBox (MappingBox, Derivable_Matrix):
    # for 2d SQUARE IMAGES ONLY
    # after first initializing, run obj.apply_mapping() immediately!
    def __init__(self, input_matrix, mapping_matrix, stride=1, bias=0):
        # prod_aggregation = lambda lst: (lst[0]* prod_aggregation(lst[1:])) if len(lst) > 1 else lst[-1]
        super(FiltBox, self).__init__(input_matrix, mapping_matrix, FiltBox_Helper_Summer(), stride);
        self.set_bias(bias)
    def set_bias(self, bias):
        self.bias = bias;
    def drv_matrix2(self, orig, mapping, output):
        if orig == None:
            orig = self.orig
        if mapping == None:
            mapping = self.mapping
        if output == None:
            output = self.output

        entry_type = type(mapping.vr[0].headc[0])
        
        map_height = len(mapping.vr)
        map_width = len(mapping.vcl)
        output_height = len(output.vr)
        output_width = len(output.vcl)
        
        drv_array = [[(entry_type)(0) for _ in range(map_width)] for _ in range(map_height)]
        drv_matrix = Matrix(drv_array)
        
        map_array = [[(entry_type)(0) for _ in range(map_width)] for _ in range(map_height)]
        map_matrix = Matrix(map_array)
        for i in range(map_height):
            for j in range(map_width):
                temp_array = [[(entry_type)(0) for _ in range(output_width)]
                              for _ in range(output_height)]
                temp_matrix = Matrix(temp_array)
                
                map_matrix.rows[i][j].set_value(1)
                map_matrix.vr[i].headc[j].set_value(1)
                # print("map_matrix")
                # print(map_matrix) # valid
                base_matrix = self.apply_mapping(self.orig, map_matrix, temp_matrix)
                # print("base matrix")
                # print(base_matrix)
                drv_sum = (entry_type)(0)
                for row in base_matrix.vr:
                    for pixel in row.headc:
                        drv_sum = drv_sum.__add__(pixel)
                # print(drv_sum)
                drv_matrix.rows[i][j] = (drv_sum)
                drv_matrix.vr[i].headc[j] = (drv_sum)
                
                map_matrix.rows[i][j].set_value(0)
                map_matrix.vr[i].headc[j].set_value(0)
        return drv_matrix
    def drv_matrix(self, input_matrix=None):
        # input_matrix in this case, is the NEXT Layer gradient matrix
        # "https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c"
        # "https://miro.medium.com/max/1400/1*Be-DgX7wlxV5bMx1vtZVAg.png"
        if input_matrix == None:
            input_matrix = self.NEXT_LAYER_GRADIENT # see link in doccomment
        new_matrix = self.output.__deepcopy__()
        grad_mappingbox = FiltBox(self.orig, input_matrix, 1)
        grad_mappingbox.finalize()
        return grad_mappingbox.output

class PoolBox(MappingBox, Derivable_Matrix):
    # for 2d SQUARE IMAGES ONLY
    # after first initializing, run obj.apply_mapping() immediately!
    def __init__(self, input_matrix, mapping_matrix = Matrix.IdentityMatrix(1), stride=1):
        # you put in a placeholder identity matrix, but we are really focusing on the dimension
        mapping_matrix_dim = len(mapping_matrix.rows)
        mapping_matrix_dim_criteria = [type(mapping_matrix_dim) == int,
                                       mapping_matrix_dim > 0,
                                       mapping_matrix_dim < len(input_matrix.rows)]
        assert all(mapping_matrix_dim_criteria)
        entry_type = type(input_matrix.vr[0].headc[0])
        placeholder_array = [[(entry_type)(1) for i in range(mapping_matrix_dim)]
                                      for j in range(mapping_matrix_dim)]
        placeholder_matrix = Matrix(placeholder_array)
        super(PoolBox, self).__init__(input_matrix, placeholder_matrix, PoolBox_Helper_Summer(), stride)
    def drv_matrix(self, input_matrix=None):
        if input_matrix == None:
            input_matrix = self.orig # revisit
        new_array = []
        entry_type = type(input_matrix.vr[0].headc[0])
        for i in range(len(input_matrix.vr)):
            new_row = []
            for j in range(len(input_matrix.vr[0].headc)):
                max_entry = input_matrix.vr[i].headc[j]
                if [j,i] in [x for x in self.aggregation.max_entries]:
                    new_row.append((entry_type)(1))
                else:
                    new_row.append((entry_type)(0))
            new_array.append(new_row)
        new_matrix = Matrix(new_array)
        return new_matrix
class ReluBox (MappingBox, Derivable_Matrix):
    def __init__(self, input_matrix, mapping_matrix = None, stride = 1):
        # entry_type = type(input_matrix.vr[0].headc[0])
        placeholder_array = [[1]] # 1 by 1 mapping box
        placeholder_matrix = Matrix(placeholder_array)
        super(ReluBox, self).__init__(input_matrix, placeholder_matrix, ReluBox_Helper_Summer(), stride=1)
    def drv_matrix(self, input_matrix=None):
        if input_matrix == None:
            input_matrix = self.output # revisit
        new_array = []
        for i in range(len(input_matrix.vr)):
            new_row = []
            for j in range(len(input_matrix.vcl)):
                # print(input_matrix.vr[i].headc[j] )
                if input_matrix.vr[i].headc[j] > 0:
                    # print("here")
                    new_row.append(Pixel(1))
                else:
                    new_row.append(Pixel(0))
            new_array.append(new_row)
        new_matrix = Matrix(new_array)
        return new_matrix
class BiasBox (MappingBox, Derivable_Matrix):
    def __init__(self, input_matrix, mapping_matrix):
        # mapping_matrix is the add matrix
        super(BiasBox, self).__init__(input_matrix,
                                      mapping_matrix,
                                      BiasBox_Helper_Summer(),
                                      1)
    def drv_matrix(self, input_matrix):
        # input matrix is the next stage chain rule (we'll need it for the dimensiona)
        next_grad_matrix = input_matrix
        entry_type = type(next_grad_matrix.vr[0].headc[0])
        next_grad_matrix_height = len(next_grad_matrix.vr)
        next_grad_matrix_width = len(next_grad_matrix.vcl)
        placeholder_array = [[(entry_type)(1) for i in range(next_grad_matrix_height)]
                                      for j in range(next_grad_matrix_width)]
        return Matrix(placeholder_array)
class Mapping_Aggregator_Function_Holder:
    def __init__(self): pass
    def apply(self, list_obj): pass
class FiltBox_Helper_Summer (Mapping_Aggregator_Function_Holder):
    def __init__(self):
        super(FiltBox_Helper_Summer, self).__init__()
    def apply(self, list_obj):
        entry_type = type(list_obj[0])
        summed = (entry_type)(0)
        for x in list_obj:
            summed = summed.__add__(x);
        return summed
class PoolBox_Helper_Summer (Mapping_Aggregator_Function_Holder):
    def __init__(self):
        super(PoolBox_Helper_Summer, self).__init__()
        self.set_max_entries([]);
        self.set_max_indices([]);
    def apply(self, list_obj):
        index = 0
        max_value = list_obj[0]
        # print("---")
        # print("list_obj is" + str(list_obj))
        for i in range(0, len(list_obj)):
            x = list_obj[i];
            if x > max_value:
                max_value = x
                index = i;
        self.add_max_index(index)
        # print("max_value is " + str(max_value))
        # print("max_indices" + str(self.max_indices))
        # print(" ")
        return max_value
    def set_max_indices(self, max_indices):
        self.max_indices = max_indices
    def add_max_index(self, max_index):
        self.max_indices.append(max_index)
    def set_max_entries(self, max_entries):
        self.max_entries = max_entries
    def add_max_entry(self, max_entry):
        print("added " + str(id(max_entry)))
        self.max_entries.append(max_entry)
class ReluBox_Helper_Summer (Mapping_Aggregator_Function_Holder):
    def __init__(self):
        super(ReluBox_Helper_Summer, self).__init__()
    def apply(self, list_obj):
        assert len(list_obj) == 1
        entry_type = type(list_obj[0])
        if list_obj[0] <= 0:
            return (entry_type)(0);
        else:
            return list_obj[0]
class BiasBox_Helper_Summer (Mapping_Aggregator_Function_Holder):
    def __init__(self):
        super(BiasBox_Helper_Summer, self).__init__()
        self.set_mapping(None)
    def set_mapping(self, bias_matrix):
        self.mapping = bias_matrix
    def apply(self, list_obj):
        assert self.mapping != None
        entry_type = type(list_obj[0])
        if list_obj[0] <= 0:
            return (entry_type)(0);
        else:
            return list_obj[0]
def tester_mapping():
    def test_filtbox():
        Am = ([[1, 1, 1, 0, 0],[0, 1, 1, 1, 0], [0, 0, 1, 1, 1],[0, 0, 1, 1, 0],[0, 1, 1, 0, 0],])
        bm = ([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
        A = Matrix(Am)
        b = Matrix(bm)
        C = FiltBox(A, b, 1, 1)
        C.finalize()
        assert C.output.rows == [[4, 3, 4], [2, 4, 3], [2, 3, 4]]
        assert C.output_dim == 3
        # assert (C.aggregation is max)
        print("filtbox successful")
        return True;
    test_filtbox()
    def test_filtbox0():
        Am = ([[1, 1, 1, 0, 0],[0, 1, 1, 1, 0], [0, 0, 1, 1, 1],[0, 0, 1, 1, 0],[0, 1, 1, 0, 0],])
        bm = ([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
        A = Matrix([[Pixel(x) for x in row] for row in Am])
        b = Matrix([[Pixel(x) for x in row] for row in bm])
        C = FiltBox(A, b, 1, 1)
        C.finalize()
        tm = [[4, 3, 4], [2, 4, 3], [2, 3, 4]]
        print(C.output.rows)
        assert C.output_dim == 3
        print("compare")
        print(str(tm))
        print(str(C.output.rows))
        print(str(C.output))
        print("filtbox0 successful")
        return True;
    test_filtbox0()
    def test_poolbox():
        print("POOLBOX TEST")
        Am = ([
            [1, 2, 1, 4],
            [0, 0, 3, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0]
        ])
        A = Matrix(Am)
        C = PoolBox(A, Matrix.IdentityMatrix(2), 2)
        C.finalize()
        try:
            assert C.output.rows == [[2, 4], [2, 0]]
            print("poolbox successful")
            return True
        except Exception as e:
            print("poolbox FAIL")
            print([[2, 4], [2, 0]])
            print(C.output.rows)
            print(" done diagnose poolbox")
            return False
    test_poolbox()
    def test_poolbox1():
        Am = ([
            [1, 2, 1, 4],
            [0, 0, 3, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0]
        ])
        Am = ([[Pixel(x) for x in row] for row in Am])
        A = Matrix(Am)
        C = PoolBox(A, Matrix.IdentityMatrix(2), 2)
        C.finalize()
        print("compare")
        print("[[2, 4], [2, 0]]")
        co_str = "[[P2 P4]\n [P2 P0]]"
        print(str(C.output.rows))
        print(str(C.output))
        assert str(C.output) == co_str
        print("poolbox1 successful")
        return True

    test_poolbox1()
    def test_relubox():
        Am = ([[-1, 2],
               [-3, 5]])
        Am = ([[Pixel(x) for x in row] for row in Am])
        A = Matrix(Am)
        C = ReluBox(A)
        C.finalize()
        # print(C.output.rows)
        co = [[0, 2], [0, 5]]
        co_str = "[[P0 P2]\n [P0 P5]]"
        print("compare")
        print(str(co))
        print(str(C.output.rows))
        assert str(C.output) == co_str
        print(str(C.output))
        print("relubox successful")
        return True;
    test_relubox()
def tester_drv_matrix():
    def test_drv_filtbox():
        Am = ([[1, 1, 1],
               [0, 1, 1],
               [0, 1, 0]])
        bm = ([[0, 1],[1, 1]])
        # A = Matrix(Am)
        # b = Matrix(bm)
        A = Matrix([[Pixel(x) for x in row] for row in Am])
        b = Matrix([[Pixel(x) for x in row] for row in bm])
        C = FiltBox(A, b, 1, 1)
        C.finalize()
        # lets pretend that next layer gradient matrix is J
        J = Matrix([[Pixel(100), Pixel(100)],[Pixel(100), Pixel(100)]])
        d = "[[P300 P400]\n [P200 P300]]"
        
        em = C.drv_matrix(J)
        # print(C.output)
        assert str(em) == d
        print("drv filtbox successful")
        return True;
    test_drv_filtbox()
    def test_drv_poolbox():
        Am = ([
            [2, 2, 1, 4],
            [1, 0, 6, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0]
        ])
        Am = ([[Pixel(x) for x in row] for row in Am])
        A = Matrix(Am)
        C = PoolBox(A, Matrix.IdentityMatrix(2), 2)
        C.finalize()
        co_str = "[[P2 P6]\n [P2 P0]]"
        assert str(C.output) == co_str
        em = C.drv_matrix(A)
        dm = [[1,0,0,0],[0,0,1,0],[0,1,1,0],[0,0,0,0]]
        d = Matrix([[Pixel(x) for x in row] for row in dm])
        print(em)
        print(d)
        assert str(em) == str(d)
        print("poolbox filtbox successful")
        return True;
    test_drv_poolbox()
    def test_drv_relubox():
        Am = ([
            [-2, -2, -1, -4],
            [1, 0, 6, 0],
            [1, 2, 0, 0],
            [0, 0, 0, 0]
        ])
        A = Matrix([[Pixel(x) for x in row] for row in Am])
        C = ReluBox(A)
        C.finalize()
        em = C.drv_matrix(A)
        dm = [[0,0,0,0],[1,0,1,0],[1,1,0,0],[0,0,0,0]]
        d = Matrix([[Pixel(x) for x in row] for row in dm])
        # print(em)
        assert str(em) == str(d)
        print("relubox filtbox successful")
        return True;
    test_drv_relubox()
    print("Differentiable resovoirs flooded")
    return True
tester_mapping()      
tester_drv_matrix()
Matrix([[Pixel(1),Pixel(2)],[Pixel(3),Pixel(4)]]).__deepcopy__()
# Note: don't ever use the .vcl attribute of the matrix class, because we never made it update
