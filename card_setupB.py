import math as mt
import numpy as np
import os
import imageio as io
from main_convolverB import *

# poland = 'https://image.shutterstock.com/image-vector/square-polish-flag-260nw-538006342.jpg'
# photo = io.imread(poland)
# print(photo.shape)

class File:
    def __init__(self, homepath, name):
        self.homepath = homepath
        self.name = name
        self.path = homepath + "/" + name
        self.filepath, self.filetype = os.path.splitext(name)

class Image(File):
    def __init__(self, document_folder_path, name):
        "conv{0} ({1}), where 0 = 1 or 0, and 1 = number of creation"
        super().__init__(document_folder_path, name)
        group = self.name[4] # 5th letter
        if group == "0":
            self.category = 0
            self.target = [0, 1]
        else:
            group == "1"
            self.category = 1
            self.target = [1, 0]

class Observation:
    # numpy
    def image_reshape(self, np_array, entry_type = float, color = False):
        np_array_shape = np_array.shape
        new_array = []
        if color == True:
            depth = np_array_shape[2]
        else:
            depth = 1
        for i in range(depth):
            new_matrix = []
            for j in range(np_array_shape[0]):
                new_row = []
                for k in range(np_array_shape[1]):
                    entry = np_array[j][k][i]
                    adjusted_entry = (-1) * (1/127.5) * (entry - 127.5)
                    new_row.append((entry_type)(adjusted_entry))
                new_matrix.append(new_row)# = np_array[i][j][0]
            new_array.append(new_matrix)
        # print(np_array)
        return new_array
    def __init__(self, information, target):
        self.set_raw_info(information)
        self.set_info(information)

        self.set_target(target)
    def set_raw_info(self, raw_info):

        self.raw_info = raw_info
    def set_info(self, info):

        self.info = self.get_image_info(info)

    def get_image_info(self, image=None):
        # returns a Tensor_3D obj of Pixel
        color = True
        if image == None:
            image = self.raw_info
        assert type(image) == Image
        image_np_array = io.imread(image.path)
        # print(image.name)
        # print(image_np_array)
        image_array = self.image_reshape(np_array=image_np_array, color=color)
        image_np_array = np.array(image_array)
        print(image_np_array)

        image_tensor = Tensor_3D(image_array)
        return image_tensor.convert_entry_type(Pixel)
    def set_target(self, target):
        self.target = target

class Dataset:  # for CONVNET
    def __init__(self, list_observations, num_categories):
        # list_observations is a List<Observation>
        self.set_list_observations(list_observations)
        self.set_size(len(list_observations))
        self.set_train_dataset(None)  # takes in dataset object
        self.set_test_dataset(None)  # takes in dataset object
        self.set_num_categories(num_categories)
    def set_num_categories(self, num_categories):
        self.num_categories = num_categories
    def set_size(self, size):
        self.size = size
    def get_size(self): # NEEDED BY CROSS_ENTROPY
        return self.size
    def set_list_observations(self, list_observations):
        self.list_observations = list_observations
    def set_num_categories(self, num_categories):
        self.num_categories = num_categories
    def set_train_dataset(self, train_dataset):
        self.train_dataset = train_dataset
    def set_test_dataset(self, test_dataset):
        self.test_dataset = test_dataset
    def split_dataset(self, list_observations = None, seed = None):
        if list_observations == None:
            list_observations = self.list_observations
        if seed != None:
            assert type(seed) == int
            assert seed >= 0
            np.random.seed(seed)

        np.random.shuffle(list_observations)

        divider = self.get_size() // 2
        train_list_observation = list_observations[:divider]
        test_list_observation = list_observations[divider:]

        train_dataset_obj = Dataset(train_list_observation, self.num_categories)
        test_dataset_obj = Dataset(test_list_observation, self.num_categories)

        self.set_train_dataset(train_dataset_obj)
        self.set_test_dataset(test_dataset_obj)

folderpath = "C:/Users/Benson/Desktop/BootlegTensorFlowFolder/convnet_images"
list_observations = []
for filename in os.listdir(folderpath):
    if filename.split(".")[-1] in ["jpeg", "jpg"]:
        print(filename)
        image = Image(folderpath, filename)
        print(image.target)
        observation = Observation(image, image.target)
        list_observations.append(observation)

Ex_Dataset = Dataset(list_observations = list_observations, num_categories= 2)
Ex_Dataset.split_dataset()





pfcs = [pfc, pfc2]

numfilts = 2
bias_matrix_stack = []
for i in range(0, numfilts):
    temp_bias_matrix = make_bias_matrix(4, 4, i+1)
    bias_matrix_stack.append(temp_bias_matrix)
bias_tensor = Tensor_3D(bias_matrix_stack)
btl = [bias_tensor]

sl = [1, 1]

def create_magnum():
    magnum = Categ_NN()
    l0 = Layer(layer_rank=0,
               network=magnum,
               weights=Matrix([[1 for _ in range(8)],
                               [1 for _ in range(8)]]),
               biases=Matrix([[1], [1]]),
               inputs=Matrix([[1] for _ in range(8)]),
               activation=SOFTMAX()
               )
    a1 = l0.forward()
    l0.a1 = a1

    target = Matrix([[0], [1]])
    ll = Loss(a0=a1, network=magnum, freq_target=target,
              loss_function=CROSSENTROPY())
    l0.next_layer = ll
    a2 = ll.forward(a1, target)
    ll.a1 = a2
    return magnum
categ_nn_obj = create_magnum()

Ex_ConvNet = ConvNet(dataset = Ex_Dataset)
Ex_ConvNet.set_categ_nn(categ_nn_obj)

sample_image_tensor = Ex_Dataset.list_observations[0].info
sample_image_tensor = ptc_fiver

L0 = Convolving_Tensor_Layer(layer_rank = 0,
                             network = Ex_ConvNet,
                             orig_tensor = sample_image_tensor,
                             weight_tensors_list=pfcs,
                             bias_tensor_list=btl,
                             strides_list=sl)
finmapper0 = L0.run_layer()
pdl = [2, 2]
L1 = Pooling_Tensor_Layer(layer_rank = 1,
                             network = Ex_ConvNet,
                             orig_tensor = finmapper0.output_tensor,
                             pool_dim_list=pdl,
                             stride_list=pdl)
finmapper1 = L1.run_layer(tensor_neuron_stage=finmapper0)
assert finmapper1.output_tensor.equalto([7, 5, 6, 6, 8, 6, 7, 7])
L2 = Fully_Connected_Vector_Layer(layer_rank = 2,
                             network = Ex_ConvNet,
                             orig_tensor = finmapper1.output_tensor)
L2.set_target_matrix(Matrix([[1],[0]]))
finmapper2 = L2.run_layer(tensor_neuron_stage=finmapper1)
assert finmapper2.output_neurons.unwrap() == [7, 5, 6, 6, 8, 6, 7, 7]
assert Ex_ConvNet.categ_nn.all_layers[0].a0.unwrap() == [7, 5, 6, 6, 8, 6, 7, 7]
softmax_layer = Ex_ConvNet.categ_nn.all_layers[-1]
print(type(softmax_layer.activation))
assert type(softmax_layer.activation) == SOFTMAX
Ex_ConvNet.CGD1_get_all_gradient_parameters()
# Ex_ConvNet.update_all_gradient_parameters()

print("end")
