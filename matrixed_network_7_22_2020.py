#!/usr/bin/env python
# coding: utf-8

# In[1516]:


import numpy as np

class Point:
    def placeholder(lst, index):
        if index >= len(lst):
            return None
        else:
            return lst[index]
    #def area3points(PointA, PointB, PointC):

    def __init__(self, coords):
        #placeholder = lambda lst, i: None if (i >= len(lst)) else lst[i]
        self.x = Point.placeholder(coords, 0)
        self.y = Point.placeholder(coords, 1)
        self.z = Point.placeholder(coords, 2)
        tempcoords = [self.x, self.y, self.z]
        self.coords = tuple([n for n in tempcoords if n != None])
        self.dimens = len([n for n in tempcoords if n != None])
    def __add__(self, otherPoint):
        assert self.dimens == otherPoint.dimens
        newcoords = [(i + j) for i, j in zip(self.coords, otherPoint.coords)]
        return Point(newcoords);
    def __mul__(self, scalar):
        assert type(scalar) == int
        newcoords = [(i * scalar) for i in self.coords]
        return Point(newcoords);
    def __sub__(self, otherPoint):
        assert self.dimens == otherPoint.dimens
        otherPoint *= -1
        newcoords = [(i + j) for i, j in zip(self.coords, otherPoint.coords)]
        return Point(newcoords);
    def __repr__(self):
        return str(self.coords)


# In[1517]:


class Vector: # STRICTLY FOR MATRIXED_NETWORK PROJECT

    def pythag(hcoords, tcoords):
        # DONT DELETE; referenced in VectorGetNorm
        tempdists = [h-c for h, c in zip(hcoords, tcoords)]
        return (sum([i**2 for i in tempdists]))**(1/2)
    def placeholder(lst, index):
        if index >= len(lst):
            return None
        else:
            return lst[index]
    def VectorGetNorm(given_Vector):
        result = Vector.pythag(given_Vector.headc, given_Vector.tailc)
        return result
    def i():
        return Vector([1, 0, 0])
    def j():
        return Vector([0, 1, 0])
    def k():
        return Vector([0, 0, 1])
    def sbl():
        return [Vector.i(), Vector.j(), Vector.k()]

    def __init__(self, inpt):
        if type(inpt) == Point:
            Vector(inpt.coords)
        elif type(inpt) in [list, tuple]: #is list or tuple
            if any((type(i) in [int, float]) for i in inpt): # 1 list
                self.headp = Point(inpt)
                self.headc = list(inpt)
                zerovect = [0 for _ in range (len(inpt))]
                self.tailp = Point(zerovect)
                self.tailc = tuple(zerovect)

                self.dimens = len(inpt)
                self.length = self.magnitude = self.norm = Vector.VectorGetNorm(self)
                self.worldpoints = [self.tailp, self.headp]

            elif any((type(i) == Point) for i in inpt): # 2 points
                self.headp = inpt[0]
                self.headc = list(inpt[0].coords)

                self.tailp = inpt[1]
                self.tailc = tuple(zerovect)
                self.dimens = len(inpt)
                self.length = self.magnitude = self.norm = Vector.VectorGetNorm(self)
                self.worldpoints = [self.tailp, self.headp]
            else:
                # vector is of some class that has a numerical value to it
                self.headc = list(inpt)
                self.dimens = len(inpt)

    def __add__(self, otherVector):
        assert type(otherVector) == Vector
        if (self.dimens == otherVector.dimens):
            temp_add_list = [(s+o) for s, o in zip(self.headc, otherVector.headc)]
            return Vector(temp_add_list)
    def __mul__(self, Scalar):
        # assert (type(Scalar) in [int, float])
        return Vector([(Scalar * i) for i in self.headc])
    def __sub__(self, otherVector):
        assert type(otherVector) == Vector
        if (self.dimens == otherVector.dimens):
            temp_add_list = [(s-o) for s, o in zip(self.headc, otherVector.headc)]
            return Vector(temp_add_list)
    def dotprod(self, otherVector):
        # print(self, otherVector)
        assert type(otherVector) == Vector
        temp_dot = 0
        for i in range(len(self.headc)):
            s = self.headc[i]
            o = otherVector.headc[i]
            # print(s, o)
            comp_prod = s * o
            # print( temp_dot , (comp_prod))

            temp_dot = temp_dot + comp_prod
        # print(temp_dot)
        return temp_dot
    def crossprod(self, otherVector):
        assert otherVector.dimens == self.dimens
        if self.dimens == 2:
            """technically not crossproduct since 
               crossproducts can only be done with 
               2 3D vectors, not 2 2D vectors"""
            return Matrix.determinant2D()
        elif self.dimens == 3:
            vl = Matrix.determinant([
                Vector.sbl(),
                self.headc,
                otherVector.headc
            ])
            return vl[0]+vl[1]+vl[2]

    def angle(self, otherVector):
        """
        (q * r)/(||q|| * ||r||)
        """
        upper = (self.dotprod(otherVector))
        otherVectorNorm = (Vector.VectorGetNorm(otherVector))
        lower = self.norm * otherVectorNorm
        cosine = upper/lower
        print(cosine)
        return mt.acos(cosine)
    def projection(self, otherVector):
        #aka parallel upon
        upper = self.dotprod(otherVector)
        print(upper)
        lower = (otherVector).dotprod(otherVector)
        print(lower)
        quotient = upper/lower
        outerVector = otherVector
        return outerVector.__mul__(quotient)
    def PArea(self, otherVector):
        temp_self_norm = Vector.VectorGetNorm(self)
        # print(temp_self_norm)
        temp_other_norm = Vector.VectorGetNorm(otherVector)
        # print(temp_other_norm)
        temp_angle = self.angle(otherVector)
        return temp_self_norm * temp_other_norm * mt.sin(temp_angle)

    def pointer_preserved_change(self, newVector):
        if type(newVector.headc[0]) in [int, float]:
            for i in range(len(self.headc)):
                self.headc[i] = newVector.headc[i]
        elif type(newVector.headc[0]) == Pixel:  # or any class with set_value
            for i in range(len(self.headc)):
                self.headc[i].set_value(newVector.headc[i].value)
    def __copy__(self):
        return Vector(self.headc)
    def __deepcopy__(self):
        if type(self.headc[0]) in [int,float,str,bool]:
            return Vector([x for x in self.headc])
        else:
            # this is assuming x is of a class. No functions, no lists, dicts,sets, tuples, 
            return Vector([x.__deepcopy__() for x in self.headc])
    def __getitem__(self, index):
        return self.headc[index]
    def __len__(self):
        return len(self.headc)
    def __sum__(self):
        entry_type = type(self.headc[0])
        total = (entry_type)(0)
        for x in self.headc:
            total = total.__add__(x)
        return total
    def __repr__(self):
        return "<{0}>".format(str(self.headc)[1:-1])


# In[1518]:


class Matrix:  # NO EQUATIONs,meant for Neural Network Project
    def IdentityMatrix(dimension):
        diagonalBoolInt = lambda x, y: 1 if x == y else 0;
        newArray = [[(0 + (i == j)) for i in range(dimension)] for j in range(dimension)]
        return Matrix(newArray)

    def determinant(grid):
        def minor(h, v, grid):
            elim_rows = [grid[row_i] for row_i in range(len(grid)) if row_i != v]
            elim_cols = [[row[x_i]
                          for x_i in range(len(grid))
                          if x_i != h]
                         for row in elim_rows]
            return elim_cols

        assert len(grid[0]) == len(grid)

        determGridLength = len(grid[0])

        if determGridLength == 2:
            return (grid[0][0] * grid[1][1]) - (grid[0][1] * grid[1][0])

        elif determGridLength > 2:
            final_list = []
            for top_i in range(len(grid[0])):
                top_x = grid[0][top_i]
                innergrid = minor(top_i, 0, grid)
                innergrid_value = Matrix.determinant(innergrid)
                alternating_sign_controller = ((-1) ** (top_i + 2))
                final_list_item = (top_x).__mul__(innergrid_value * alternating_sign_controller)
                final_list.append(final_list_item)
            return final_list

    def __init__(self, array):
        assert type(array) == list
        if type(array[0]) == list:
            self.rows = array;
            self.vr = [Vector(row) for row in self.rows]
            self.vcl = [Vector([row[i] for row in (self.rows)])
                for i in range(len(array[0]))]
        elif type(array[0]) == Vector:
            self.rows = [row.headc for row in array];
            self.vr = [row for row in array]
            self.vcl = [Vector([row.headc[i] for row in (array)])
                for i in range(len(array[0].headc))]

    # """
    def __add__(self, other):
        if type(other) in [float, int]:
            newArray = [(x.__add__(other)) for row in self.rows for x in row]
            return Matrix(newArray)  # """
        if type(other) == Matrix:
            assert (len(self.vr) == len(other.vr)) and (len(self.vr[0]) == len(other.vr[0]))
            newArray = []
            for i in range(len(other.vr)):
                vectA = self.vr[i]
                vectB = other.vr[i]
                vect_added = vectA + vectB
                newRow = vect_added.headc
                newArray.append(newRow)
            return Matrix(newArray)  # """
    def __sub__(self, other):
        if type(other) in [float, int]:
            newArray = [(x.__sub__(other)) for row in self.vr for x in row.headc]
            return Matrix(newArray)  # """
        if type(other) == Matrix:
            assert (len(self.vr) == len(other.vr)) and (len(self.vr[0]) == len(other.vr[0]))
            newArray = []
            for i in range(len(other.vr)):
                vectA = self.vr[i]
                vectB = other.vr[i]
                vect_added = vectA - vectB
                newRow = vect_added.headc
                newArray.append(newRow)
            return Matrix(newArray)  # """
    def hadamard(self, other):
        if type(other) == Matrix:
            # assert (len(self.vr) == len(other.vr)) and (len(self.vcl) == len(other.vcl))
            newArray = []
            for i in range(len(self.vr)):
                newRow = []
                for j in range(len(other.vr[0].headc)):
                    value = self.vr[i].headc[j] * other.vr[i].headc[j]
                    newRow.append(value)
                newArray.append(newRow)
            return Matrix(newArray)  # """

    def __mul__(self, other):
        if type(other) in [int, float]:
            newArray = []
            for row_i in range(len(self.vr)):
                newArray.append(((self.vr[row_i]).__mul__(other)).headc)
            return Matrix(newArray)

        elif type(other) == Matrix:
            emptarray = [[0 for _ in range(len(other.vr[0].headc))] for _ in range(len(self.vr))]
            # assert (len(self.vcl) == len(other.vr))
            height = len(self.vr)
            width = len(other.vr[0].headc)
            for i in range(height): # row in enumerate(self.vr):
                row_vect = self.vr[i];
                for j in range(width):
                    col_vect = Vector([row.headc[j] for row in other.vr])
                    value = (row_vect).dotprod(col_vect)
                    emptarray[i][j] = value;
            return Matrix(emptarray)
        elif type(other) == Vector:
            column_vector = Matrix([[x] for x in self.headc])
            return self.__mul__(column_vector)

    def sector(self, x_0, x_1, y_0, y_1):
        x_criteria = [(x_0 >= 0), (x_1 < len(self.rows[0])), (x_0 <= x_1)]
        y_criteria = [(y_0 >= 0), (y_1 < len(self.rows)), (y_0 <= y_1)]
        # assert (all(x_criteria) and all(y_criteria))
        sector = [row.headc[x_0: x_1 + 1] for row in self.vr[y_0: y_1 + 1]]
        # DO NOT DEEPCOPY THIS, or poolbox will fall apart

        submatrix = Matrix(sector)

        return submatrix

    def unwrap(self):
        return [x for row in self.vr for x in row.headc]
    def to_vector(self):
        return Vector([x[0] for x in self.vr])

    def __copy__(self):
        return Matrix(self.rows)
    def __deepcopy__(self):
        return Matrix([vect.__deepcopy__().headc for vect in self.vr])

    def sw(self, rowA_index, rowB_index):
        self.vr[rowA_index - 1], self.vr[rowB_index - 1] = self.vr[rowB_index - 1], self.vr[rowA_index - 1]
        return self

    def sc(self, rowA_index, scalar):
        self.vr[rowA_index - 1] *= scalar
        return self

    def sm(self, rowA_index, rowB_index, scalar=1):
        self.vr[rowA_index - 1] += (self.vr[rowB_index - 1] * scalar)
        return self
    def __sum__(self):
        entry_type = type(self.vr[0].headc[0])
        total = (entry_type)(0)
        for vector in self.vr:
            vector_sum = vector.__sum__()
            total = total.__add__(vector_sum)
        return total
    def __getitem__(self, index):
        return self.vr[index]
    def __len__(self):
        return len(self.vr)
    def get_area(self):
        # get area vs __sum__
        # __sum__ adds up all of the items in the matrix.
        # while get_area counts the amount of items in the matrix
        # get_area returns an int x where x >= 0
        area = 0
        for vector in self.vr:
            area += len(vector.headc)
        return area
    def __repr__(self):
        grid = self.vr

        return str(np.array([row.headc for row in grid]))
        # "grid" must be self.vr for this, not self.rows, since the row operations use self.vr,
        # we can see the updated matrix. If we used self.rows, it will show the original matrix only
    def dilate(self, d=0):
        mw = len(self.vr)
        assert mw > 1
        entry_type = type(self.vr[0].headc[0])
        nmw = mw + (d * (mw - 1))
        nm = [[(entry_type)(0) for _ in range(nmw)] for _ in range(nmw)]
        for i in range(0, nmw, d+1):
            for j in range(0, nmw, d+1):
                nm[i][j] = (self.vr[i // (d + 1)].headc[j // (d + 1)])
        return Matrix(nm)
    def pad(self, padn = 0):
        assert padn >= 0
        entry_type = type(self.vr[0].headc[0])
        mw = len(self.vr)
        nmw = mw + (2 * padn)
        nm = [[(entry_type)(0) for _ in range(nmw)] for _ in range(nmw)]
        for i in range(padn, padn + mw):
            for j in range(padn, padn + mw):
                nm[i][j] = self.vr[i - padn].headc[j - padn]
        return Matrix(nm)
    def rotate(self):
        # rotate clockwise by 90 degrees
        arrayed = [[row.headc] for row in self.vr]
        #assert len(array) == len(array[0])
        width = len(arrayed[0])
        sideways_array = [[row[i] for row in arrayed][::-1]
                          for i in range(width)]
        return Matrix(sideways_array)
    def pointer_preserved_change(self, newMatrix):
        assert type(self.vr[0]) == Vector
        assert type(newMatrix.vr[0]) == Vector
        if len(newMatrix.vr) != len(self.vr):
            print("ppc wrong in dimension")
            print(newMatrix)
            print(" ")
            print(self)
            raise AssertionError()
        for i in range(len(self.vr)):
            self.vr[i].pointer_preserved_change(newMatrix.vr[i])


# In[1519]:


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
class CROSSENTROPY (COST_FUNCTION_HOLDER):
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
            temp_log_loss_drv = t * mt.log(p + 1e-8)
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

            temp_log_loss_drv = (t * (1/(p + 1e-8))) * (-1)
            sum_log_loss_drv.append([temp_log_loss_drv])
        return Matrix(sum_log_loss_drv)


# In[1520]:


class ACTIVATION_FUNCTION_HOLDER:
    def __init__(self):
        pass
    def apply(self, x):
        pass
    def apply_drv(self, x):
        pass


# In[1521]:


import math as mt
class SIGMOID (ACTIVATION_FUNCTION_HOLDER):
    def __init__(self):
        super(SIGMOID, self).__init__()
    def apply(self, m):
        nm = []
        for v in m.vr:
            x = v[0]
            nv = (1) / (1 + ((mt.e) ** (-x)))
            nm.append([nv])
        return Matrix(nm)
    def apply_drv(self, m):
        nm = []
        for v in m.vr:
            x = v[0]
            nv0 = (1) / (1 + ((mt.e) ** (-x)))
            nv = nv0 * (1 - nv0)
            nm.append([nv])
        return Matrix(nm)
class SOFTMAX (ACTIVATION_FUNCTION_HOLDER):
    # before the cross_entropy cost function,- there is the softmax activation layer
    # and before the softmax activation layer- there is the logits layer
    def __init__(self):
        super(SOFTMAX, self).__init__()
    def apply(self, matrix):
        # column vector input
        summed_averager = 0
        for v in matrix.vr:
            x = v[0]
            summed_averager += mt.e**(x)
        softmax_column = []
        for v in matrix.vr:
            x = v[0]
            softmax_column.append([mt.e**(x)/summed_averager])
        total_a0 = sum([x[0] for x in softmax_column])
        if not (int(round(total_a0)) == 1): # > 0.9 and total_a0 <= 1.0): # adds up to 1 or is close
            raise AssertionError(str(total_a0) + " not close to 1")
        return Matrix(softmax_column)

    def apply_drv(self,matrix):
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


# In[1522]:


import numpy as np
class Categ_NN:
    def __init__(self):
        self.learning_rate = 0.1
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
        learning_gradient = [(part_drv * self.learning_rate) for part_dev in gradient]
        new_parameters = []
        for i in range(len(self.all_parameters)):
            parameter = self.all_parameters[i]
            part_drv = learning_gradient[i]
            diffrence = parameter - part_drv
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
        for temp_layer in self.all_layers[:-1]:
            temp_layer.post_creation_organize()
        output_layer = self.all_layers[-1]
        output_layer.post_creation_organize_output()
        self.final_loss.post_creation_organize()
    
    def reconfigure_observation(self, input_, target):
        # updates the input and output
        self.all_layers[0].a0.pointer_preserved_change(input_)
        self.final_loss.freq_target.pointer_preserved_change(target)
        
    # FUNCTIONS FOR TRAINING
    def run_batches(self, train_datalist, num_batches, network_obj = None):
        if network_obj == None:
            network_obj = self
        # network_obj with layers and loss already set to default values
        # train_datalist =  list of obsverations objects, scrambled
        # num_batches = int number of batch runs you want
        batch_len = len(train_datalist) // num_batches # length (num of observations) per batch

        batch_run_i = 0
        imported_parameters = network_obj._rand_parameters(network_obj.all_parameters)
        while batch_run_i < num_batches:
            start = (batch_len * batch_run_i)
            end = (batch_len * batch_run_i) + batch_len
            train_sublist = train_datalist[start:end]
            print("BATCH {0}/{1}".format(batch_run_i, num_batches))
            imported_parameters = self._average_gradient(batch_observations=train_sublist, 
                                                        imported_parameters=imported_parameters,
                                                        network_obj = None)
            # print(imported_parameters)
            batch_run_i += 1
        return imported_parameters
    def _average_gradient(self, batch_observations=None, imported_parameters=None, network_obj = None):
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
        list_gradients = [batch_parameters]

        # STEP 2: summing gradients in batch
        batch_i = 1 # start from second index
        while batch_i < batch_n:

            # change inputs, target
            temp_observation = batch_observations[batch_i]
            input_ = temp_observation.info
            target = temp_observation.target
            network_obj.reconfigure_observation(input_, target)

            # paremeters need not be changed, already updated in the first batch observation
            # and each osbervation of batch is recycling the same network obj
            network_obj.GD4_update_reconfigurations()
            temp_grad = network_obj.GD1_create_gradient()

            # collect gradient # aka adding to batch_parameters
            list_gradients.append(temp_grad)
            for j in range(len(temp_grad)):
                temp_part_drv = temp_grad[j]
                summed_part_drv = batch_parameters[j] + temp_part_drv
                batch_parameters[j] = summed_part_drv

            batch_i += 1
        # STEP 3: dividing each part_drv in gradient by batch_n
        inverse_summed_averager = 1/(batch_n)
        for i in range(len(batch_parameters)):
            part_drv = batch_parameters[i]
            batch_parameters[i] = part_drv * inverse_summed_averager
        # STEP 4: finalize gradient for export in next batch round
        exported_parameters = batch_parameters
        return exported_parameters # will be imported_parameters for next round
    def _rand_parameters(self, parameter_list):
        rand_parameter_list = []
        for parameter in parameter_list:
            assert type(parameter) == Matrix
            height = len(parameter.vr)
            width = len(parameter.vr[0].headc)
            np_parameter = np.random.rand(height, width)
            matrix_parameter = Matrix(np_parameter.tolist())
            rand_parameter_list.append(matrix_parameter)
        return rand_parameter_list 


# In[1523]:



class Observation:
    def __init__(self, information, target):
        self.set_raw_info(information)
        self.set_info(information)
        self.set_target(target)
    def set_raw_info(self, raw_info):
        self.raw_info = raw_info
    def set_info(self, info):
        self.info = (info) # self.get_image_info(info)
    def set_target(self, target):
        self.target = target


# In[1524]:



class ENCODER_HOLDER:
    # NOT FOR DERIVATIVES
    # returns one_hot_encoded column matrices,
    # not the function's value itself
    def __init__(self):
        pass
    def apply(self, varbs):
        pass

class REGION_ENCODER (ENCODER_HOLDER):
    # single varb circ function holder
    def __init__(self):
        super(REGION_ENCODER).__init__();
    def apply(self, varbs):
        assert len(varbs) == 2
        x = varbs[0]
        y = varbs[1]
        if y >= 30:
            return Matrix([[1], [0], [0]])
        else:
            if x <= 27:
                return Matrix([[0], [1], [0]])
            else:
                return Matrix([[0], [0], [1]])


# In[1525]:


import random

def grouping(x, y):
    if y >= 30:
        return Matrix([[1], [0], [0]])
    else:
        if x <= 27:
            return Matrix([[0], [1], [0]])
        else:
            return Matrix([[0], [0], [1]])


class Categ_NN_Executor:
    def __init__(self, categ_nn_obj):
        self.categ_nn_obj = categ_nn_obj
        self.dataset = []
        self.train_dataset = []
        self.test_dataset =[]
    def generate_dataset(self):
        observations = []
        for i in range(50):
            for j in range(50):
                group = grouping(i, j)
                observation = Observation(Matrix([[i], [j]]), group)
                observations.append(observation)
        self.dataset = observations
        return observations
    def split_dataset(self, train_proportion = None):
        if train_proportion == None:
            train_proportion = 0.8
        random.shuffle(self.dataset)
        
        dataset_len = len(self.dataset)
        split_index = int(dataset_len * 0.8)
        self.train_dataset = (self.dataset)[:split_index]
        self.test_dataset =(self.dataset)[split_index:]
    def batch_gradient_descent(self):
        descended_parameters = (self.categ_nn_obj).run_batches(
            train_datalist = self.train_dataset, 
                                        num_batches = 10)
        return descended_parameters
    def verify_test_dataset(self, final_parameters):
        self.categ_nn_obj.GD3_reconfigure_parameters(final_parameters)
        test_losses = []
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
            test_loss = loss_stage.a1
            # print("new input a0 {0}".format(str(self.categ_nn_obj.all_layers[0].a0)))
            # print("new loss a0 {0}".format(str(self.categ_nn_obj.final_loss.a0)))
            # print("new target a0 {0}".format(str(self.categ_nn_obj.final_loss.freq_target)))

            targl = trial_target.unwrap() # targ list
            predl = loss_stage.a0.__deepcopy__().unwrap()
            
            # print(" -- ")
            # print(loss_stage)
            # print(targl)
            # print(predl)
            targl_idx = max(range(len(targl)), key=targl.__getitem__)
            predl_idx = max(range(len(predl)), key=predl.__getitem__)
            # print(predl_idx, targl_idx)
            if targl_idx == predl_idx:
               successes += 1
            test_losses.append(test_loss)
        print(str(successes) + "/" + str(len(self.test_dataset)))
        return test_losses
    def main_executor(self):
        self.generate_dataset()
        self.split_dataset()
        final_parameters = self.batch_gradient_descent()
        self.verify_test_dataset(final_parameters)


# In[1526]:


class Layer:
    def __init__(self, layer_rank = 0, network = None,
                 weights = None, biases = None,
                 inputs = None, activation = None):
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
        
        ws = (W * (a0)) # ws = weighted_sum
        z = ws + (b)
        a1 = activation.apply(z)
        return a1
    
    def drv_a(self):
        # print("   MMMMMMM   ")
        # for a1 ( next layer's a0)
        # outputs a vector
        a0 = self.a0
        nl = self.next_layer
        W = self.W
        b = self.b
        activation = self.activation        
        
        #is strictly for a1, not a0
        
        # if next not loss

        z1 = (W * (a0)) + (b)
        
        if type(activation) == SOFTMAX:
            jacobian = activation.apply_drv(z1)
            assert type(nl) == Loss
            assert type(nl.loss_function) == CROSSENTROPY
            loss_drv = nl.drv_loss() # drv_column
            drv_z1 = jacobian * loss_drv # matmul
            return drv_z1
        
        drv_z1 = self.activation.apply_drv(z1)
        # print('drv_z1\n'+str(drv_z1)+'\n')        
        
        if type(nl) == Loss: # BRANCH
            # if next layer is loss, this layer is output_activation
            next_drv_a = Matrix([[nl.drv_loss()[0][0]] for _ in range(len(drv_z1))])
        else:
            next_drv_a = nl.drv_a() # drv_a2
        # print('next_drv_a\n'+str(next_drv_a)+'\n')
        a1 = nl.a0
        # print('a1\n'+str(a1)+'\n')

        # scrwm means sum_chain_rule_weight_matrix of Matrix(scrwa)
        scrwm = self._concat_drv_a(a1, N = len(next_drv_a))
        # print('scrwm\n'+str(scrwm)+'\n')
        

        chr_a1 = (scrwm) * (next_drv_a) # chain rule vector
        # print('chr_a1\n'+str(chr_a1)+'\n')
        
        drv_a1 = drv_z1.hadamard(chr_a1)
        # print('drv_a1\n'+str(drv_a1)+'\n')
        # print(" wwwwwww ")
        
        return drv_a1 #  is vectorlike

    def drv_W(self, *W):
        # outputs a matrix
        W = self.W
        a0 = self.a0
        nl = self.next_layer
        a1 = self.next_layer.a0
        
        w_height = len(W) # height of weight matrix
        dW_array = [a0.to_vector() for i in range(w_height)] #
        dW = Matrix(dW_array)
        # print('dW\n'+str(dW)+'\n')
        next_drv_a = self.drv_a()
        
        dW_width = len(dW.vr[0]) # row length
        nldm = self._concat_drv_a(next_drv_a, dW_width)
        # print('nldm\n'+str(nldm)+'\n')
        # nldm - next layer derivative matrix
        # next layer vector (Drvs) or drv_a, repeated side by side to match weight matrix
        drv_W_ = dW.hadamard(nldm)
        return drv_W_ # is matrix
    
    def drv_b(self):
        # drv_b_ = self.drv_a() # WRONG, we must do it to the next layer
        drv_b_ = self.drv_a() 
        return drv_b_ # is vectorlike
    def __repr__(self):
        str_form = """L-{0};\nA0-{1};\nB-{2};\nW-{3}\n"""
        return str_form.format(self.layer_rank, self.a0, self.b, self.W)
    def backpropagation(self):
        weights_grad = self.drv_W()
        biases_grad = self.drv_b()
        layer_gradients = [weights_grad, biases_grad]
        return layer_gradients
    def post_creation_organize(self,  next_W = None, next_b = None, next_activation = None):
        # organizes everything after when the object is first created
        # OR When the layer is being updated
        # network and layer_rank are the only attributes that must be stated when the new layer is created
        # but we must assume that a0 is given
        
        assert self.network != None # in order of priority
        assert self.a0 != None
        assert self.W != None
        assert self.b != None
        assert self.activation != None
        
        # WARNING: forward() is ran in this function
        
        # CASE 1: the object is first created:
        if self.a1 == None or self.next_layer == None:
            # next_W, next_b, next_activation for a1 must be set MANUALLY
            # either by the function's arguments or after THIS function is ran
            a1 = self.forward()
            nl  = Layer(layer_rank = self.layer_rank + 1, network = Onion,
                                weights = next_W, biases = next_b,
                                inputs = a1, activation = next_activation)
            self.a1 = a1
            self.next_layer = nl
        # CASE 2: the object is being updated:
        # please know that: a0 has already been updated recursively and AUTOMATICALLY
        # and only case where a0 isn't updated is the first layer, or inputs
        # next_W, next_b, and next_activation won't be used here since they were already reconfigured
        # but the next layer's a0 WILL BE modified (whiel preserving pointers)
        else:
            # here, the next layer exists already
            nl = self.next_layer
            new_a1 = self.forward() # with new a0 and reconfigured W and b
            self.a1.pointer_preserved_change(new_a1)
            nl.a0.pointer_preserved_change(new_a1)
    def post_creation_organize_output(self, next_target = None, next_loss_function = None):
        # FOR PENULTIMATE LAYER
        # CASE 1: the object is first created:
        if self.a1 == None or self.next_layer == None:
            
            # WARNING: forward() is ran in this function

            
            # next_W, next_b, next_activation for a1 must be set MANUALLY
            # either by the function's arguments or after THIS function is ran
            a1 = self.forward()
            nl  = Loss(a0 = self.a0, network = Onion,
                      freq_target = next_target, 
                       loss_function = next_loss_function)
            self.a1 = a1
            self.next_layer = nl
        # CASE 2: the object is being updated:
        # please know that: a0 has already been updated recursively and AUTOMATICALLY
        # and only case where a0 isn't updated is the first layer, or inputs
        # next_target, next_loss_function won't be used here since they were already reconfigured
        # but the next layer's a0 WILL BE modified (whiel preserving pointers)
        else:
            # here, the next layer exists already
            nl = self.next_layer
            new_a1 = self.forward() # with new a0 and reconfigured W and b
            self.a1.pointer_preserved_change(new_a1)
            nl.a0.pointer_preserved_change(new_a1)


# In[1527]:


class Loss:
    def __init__(self, a0 = None, network = None, freq_target = None, loss_function = None):
        self.a0 = self.output = a0 # post activation outputs before loss value
        network.final_loss = self
        self.network = network
        self.freq_target = freq_target # unmatrixed array
        self.loss_function = loss_function
        self.a1 = self.loss = None
    def forward(self, p = None,t = None):
        if p == None:
            p = self.a0
        if t == None:
            assert self.freq_target != None
            t = self.freq_target
        loss = self.loss_function.apply(p,t) # returns a matrix
        return loss
    def drv_loss(self, p = None, t = None):
        if p == None:
            p = self.a0
        if t == None:
            assert self.freq_target != None
            t = self.freq_target
        drv_loss = self.loss_function.apply_drv(p,t)
        return drv_loss
    def drv_a(self):
        p = self.a0
        pl = self.network.all_layers[-1]
        drv_z0 = pl.activation.apply_drv((pl.W*pl.a0) + pl.b)
        assert self.freq_target != None
        t = self.freq_target
        drv_loss = self.loss_function.apply_drv(p,t) # returns a matrix
        return drv_z0.hadamard(drv_loss) # is a matrix
    def post_creation_organize(self, freq_target = None):
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


# In[1528]:


Softboi = Categ_NN()
l0 = Layer(layer_rank = 0,
           network = Softboi,
           weights = Matrix([[1, 1], 
                             [1, 1]]),
           biases = Matrix([[1], [1]]),
           inputs = Matrix([[1], [1]]),
           activation = SIGMOID()
           )
a1 = l0.forward()

l1 = Layer(layer_rank = 1,
           network = Softboi,
           weights = Matrix([[1, 1], 
                             [1, 1],
                             [1, 1]]),
           biases = Matrix([[1], [1], [1]]),
           inputs = a1,
           activation = SOFTMAX()
           )
a1 = l0.forward()
l0.a1 = a1
l0.next_layer = l1
a2 = l1.forward()

target = Matrix([[0], [1], [0]])
ll = Loss(a0 = a2, network = Softboi, freq_target = target,
          loss_function = CROSSENTROPY())
a3 = ll.forward(a2, target)
l1.a1 = a2
l1.next_layer = ll
ll.a1 = a3


# In[1529]:


Softboighost = Categ_NN_Executor(Softboi)
Softboighost.main_executor()


# In[1530]:



Onion = Categ_NN()
l0 = Layer(layer_rank = 0,
           network = Onion,
           weights = Matrix([[1, 3], 
                             [2, 4]]),
           biases = Matrix([[1], [1]]),
           inputs = Matrix([[1], [2]]),
           activation = SIGMOID()
           )
a1 = l0.forward()

l1  = Layer(layer_rank = 1,
           network = Onion,
           weights = Matrix([[1, 2]]),
           biases = Matrix([[1]]),
           inputs = a1,
           activation = SIGMOID()
           )
l0.a1 = a1
l0.next_layer = l1
a2 = l1.forward()

ll = Loss(a0 = a2, network = Onion, freq_target = Matrix([[0]]),
          loss_function = MEAN_SQUARED_ERROR())
a3 = ll.forward(a2, [[0]])
l1.a1 = a2
l1.next_layer = ll
ll.a1 = a3


# In[1531]:


def grouping(x, y):
    line = (1/3 * x) + 6
    if y <= line:
        # underneath
        return Matrix([[1]])
    else:
        return Matrix([[0]])

Onionghost = Categ_NN_Executor(Onion)
Onionghost.main_executor()


# 
# Onion = Categ_NN()
# l0 = Layer(layer_rank = 0,
#            network = Onion,
#            weights = Matrix([[1, 3, 5], 
#                              [2, 4, 6]]),
#            biases = Matrix([[1], [1]]),
#            inputs = Matrix([[1], [2], [3]]),
#            activation = SIGMOID()
#            )
# a1 = l0.forward()
# 
# l1  = Layer(layer_rank = 1,
#            network = Onion,
#            weights = Matrix([[1, 2]]),
#            biases = Matrix([[1]]),
#            inputs = a1,
#            activation = SIGMOID()
#            )
# l0.a1 = a1
# l0.next_layer = l1
# a2 = l1.forward()
# 
# ll = Loss(a0 = a2, network = Onion, freq_target = [[0]],
#           loss_function = MEAN_SQUARED_ERROR())
# a3 = ll.forward(a2, [[0]])
# l1.a1 = a2
# l1.next_layer = ll
# ll.a1 = a3

# l1.a1 # should be close to 0.98201

# ll.a1 # should be close to 0.96435

# l1.drv_W()
# # [[0.0340661 0.0340661]]

# l1.drv_b() #[[0.03469004]]

# l1.drv_a() # 0.03469004
# # [[0.0340661]]
# 

# ll.drv_a() # should look like [[0.03469004]]
# # while ll.drv_loss is [[1.96402758]]

# l0.drv_a()
# 
# # [[3.49582236e-12]
# #  [8.66856594e-15]]

# l0.drv_b()

# l0.drv_W()
# # [[3.49582236e-12 6.99164472e-12 1.04874671e-11]
# #  [8.66856594e-15 1.73371319e-14 2.60056978e-14]]

# Onion.all_parameters

# Onion.all_layers[0].W# = Matrix.IdentityMatrix(2)

# (len(set([id(Onion.all_layers[0].W), 
#           id(l0.W), 
#           id(Onion.all_parameters[0])])) == 1)

# Onion.all_parameters[0] = Matrix.IdentityMatrix(2)
# l0.W, Onion.all_layers[0].W

# (Onion.all_layers[0].W).pointer_preserved_change(Matrix([[1, 0, 0],[0, 1, 0]]))

# ids = [id(Onion.all_layers[0].W), 
#           id(l0.W), 
#           id(Onion.all_parameters[0])]
# ids

# l0.W

# Onion.all_parameters[0]

# l0.W = Matrix.IdentityMatrix(2)
# l0.W

# Onion.all_parameters

# 

# Softbean = Categ_NN()
# l0 = Layer(layer_rank = 0,
#            network = Softbean,
#            weights = Matrix([[1, 3, 5], 
#                              [2, 4, 6]]),
#            biases = Matrix([[1], [1]]),
#            inputs = Matrix([[1], [2], [3]]),
#            activation = SIGMOID()
#            )
# a1 = l0.forward()
# 
# l1 = Layer(layer_rank = 1,
#            network = Softbean,
#            weights = Matrix([[1, 2], 
#                              [2, 3]]),
#            biases = Matrix([[1], [1]]),
#            inputs = a1,
#            activation = SOFTMAX()
#            )
# a1 = l0.forward()
# l0.a1 = a1
# l0.next_layer = l1
# a2 = l1.forward()
# 
# target = Matrix([[0], [1]])
# ll = Loss(a0 = a2, network = Softbean, freq_target = target,
#           loss_function = CROSSENTROPY())
# a3 = ll.forward(a2, target)
# l1.a1 = a2
# l1.next_layer = ll
# ll.a1 = a3

# l0.drv_W()
# # [[-2.84824393e-27 -5.69648786e-27 -8.54473178e-27]
# #  [-7.06277029e-30 -1.41255406e-29 -2.11883109e-29]]

# l0.drv_a()

# In[ ]:





# In[1532]:


# each layer's stages are 
# the GIVEN a0, the weight, zbias, active
# OR
# the a0, the weight, zbias, 


# height of z == w*a1 == b == a1/nl a0 == for_w
# height of a0 =/= above;let height of a0 = Q, let height of z = R

# for drv_W()  #=================================
# W_h = weight height, w_w = weight_width
# output = (W_h)*(W_w) matrix
# since it's just a0's (post activated), then
# Dw = We transpose a0, stack W_h # of clones of it
# thus it is an (R)*(Q) matrix
# that is just the local drv lol

# We now do the outside work: Make an (R)*(Q) matrix 
# by getting drv col vector drv_a1 (with drv of activation too), concat with clones Q number of times
# -- with loss --
# if next layer is loss, then 
# chr_matrix built drv_a1 (with drv of actication) # 


# for drv_a() (and drv_b):
# output = (Q)*(1) Matrix # a is for a0, or (R)*(1) if for a1
# for clarity: input = aN (a0 or a1)
# ingredients: [previous weight,aN-1,bias],A weight matrix after this, and drv col vector of drv_aN+1
# drv_z1 = drv_prev_act(prevW*An-1 + b)
# drv_z1.hadamard(??)
# ?? = new_Weight_matrix , matmul with next drv_a() (drv_A of aN+1) to make chainrule (NOT DRV) of aN,
# since drv_An is drv_z1.hadamard(??)
# let S = height of aN+1
# thus for matmul, to get (R)*1 matrix,  chr_matrix must be R*S, made with drv_aN+1, concat with clones for S times
# -- with loss --
