
import numpy as np

class Vector:  # STRICTLY FOR MATRIXED_NETWORK PROJECT

    def pythag(hcoords, tcoords):
        # DONT DELETE; referenced in VectorGetNorm
        tempdists = [h - c for h, c in zip(hcoords, tcoords)]
        return (sum([i ** 2 for i in tempdists])) ** (1 / 2)

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
        if type(inpt) in [list, tuple]:  # is list or tuple
            if any((type(i) in [int, float]) for i in inpt):  # 1 list
                self.headc = list(inpt)
                zerovect = [0 for _ in range(len(inpt))]
                self.tailc = tuple(zerovect)

                self.dimens = len(inpt)
                self.length = self.magnitude = self.norm = Vector.VectorGetNorm(self)
            else:
                # vector is of some class that has a numerical value to it
                self.headc = list(inpt)
                self.dimens = len(inpt)

    def __add__(self, otherVector):
        assert type(otherVector) == Vector
        if (self.dimens == otherVector.dimens):
            temp_add_list = [(s + o) for s, o in zip(self.headc, otherVector.headc)]
            return Vector(temp_add_list)

    def __mul__(self, Scalar):
        # assert (type(Scalar) in [int, float])
        return Vector([(Scalar * i) for i in self.headc])

    def __sub__(self, otherVector):
        assert type(otherVector) == Vector
        if (self.dimens == otherVector.dimens):
            temp_add_list = [(s - o) for s, o in zip(self.headc, otherVector.headc)]
            return Vector(temp_add_list)

    def __eq__(self, other):
        if type(other) == Vector:
            if len(self.headc) == len(other.headc):
                for i in range(len(self.headc)):
                    if self.headc[i] != other.headc[i]:
                        return False
                return True
            else:
                return False
        else:
            return False

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
            return vl[0] + vl[1] + vl[2]
    def to_matrix(self):
        return Matrix([[x] for x in self.headc])

    def pointer_preserved_change(self, newVector):

        if type(newVector.headc[0]) in [int, float]:
            for i in range(len(self.headc)):
                self.headc[i] = newVector.headc[i]
        else:  # or any class with set_value

            for i in range(len(self.headc)):
                self.headc[i].set_value(newVector.headc[i].value)

    def __copy__(self):
        return Vector(self.headc)

    def __deepcopy__(self):
        if type(self.headc[0]) in [int, float, str, bool]:
            return Vector([x for x in self.headc])
        else:
            # this is assuming x is of a class. No functions, no lists, dicts,sets, tuples,
            return Vector([x.__deepcopy__() for x in self.headc])

    def __getitem__(self, index):
        return self.headc[index]

    def __len__(self):
        return len(self.headc)

    def __sum__(self):
        

        total = 0
        for x in self.headc:
            try:
                total = total + x
            except:
                print(x)
                raise TypeError("wrong type")
        return total

    def __repr__(self):
        return "<{0}>".format(str(self.headc)[1:-1])


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
            for i in range(height):  # row in enumerate(self.vr):
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

    def transpose(self, matrix=None):
        if matrix == None:
            matrix = self
        matrix_height = len(matrix.vr)
        matrix_width_ = len(matrix.vr[0])
        new_transposition = [[0 for _ in range(matrix_height)]
                             for _ in range(matrix_width_)]
        for j in range(len(matrix.vr[0])):
            for i in range(len(matrix.vr)):
                value = matrix.vr[i].headc[j]
                new_transposition[j][i] = value
        return Matrix(new_transposition)

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
        total = 0
        for vector in self.vr:
            vector_sum = vector.__sum__()
            total = total + (vector_sum)
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

    def __eq__(self, other):
        if type(other) == Matrix:
            if len(self.vr) == len(other.vr) and len(self.vr[0]) == len(other.vr[0]):
                for i in range(len(self.vr)):
                    if not (self.vr[i] == other.vr[i]):
                        return False
                return True
            else:
                return False
        else:
            return False

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
        for i in range(0, nmw, d + 1):
            for j in range(0, nmw, d + 1):
                nm[i][j] = (self.vr[i // (d + 1)].headc[j // (d + 1)])
        return Matrix(nm)

    def pad(self, padn=0):
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
        arrayed = [row.headc for row in self.vr]
        width = len(arrayed[0])
        sideways_array = [[row[i] for row in arrayed][::-1] for i in range(width)]
        return Matrix(sideways_array)
    def flip(self):
        # flip upsidedown; just does rotate twice
        return self.rotate().rotate()

    def pointer_preserved_change(self, newMatrix):
        assert type(self.vr[0]) == Vector
        assert type(newMatrix.vr[0]) == Vector
        if len(newMatrix.vr) != len(self.vr):
            # print(newMatrix)
            # print(" ")
            # print(self)
            raise AssertionError()
        for i in range(len(self.vr)):
            self.vr[i].pointer_preserved_change(newMatrix.vr[i])


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
            newArray = [(matrix * (other)) for matrix in self.md]
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
    def __getitem__(self, index):
        return self.md[index]
    
    def __len__(self):
        return len(self.md)
    
    def get_volume(self):
        # get volume vs __sum__
        # __sum__ adds up all of the items in the tensor.
        # while get_volume counts the amount of items in the tensor
        # get_volume returns an int x where x >= 0
        
        # sum for 3d would be like, get mass
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
