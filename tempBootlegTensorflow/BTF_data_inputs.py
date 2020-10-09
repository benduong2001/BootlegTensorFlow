import os
import imageio as io
from BTF_tensor_classes import *
from BTF_global_folderpaths import *

class File:
    def __init__(self, homepath, name):
        self.homepath = homepath
        self.name = name
        self.path = homepath + "/" + name
        self.filepath, self.filetype = os.path.splitext(name)

class Image_File(File):
    def __init__(self, document_folder_path, name):
        "conv{0} ({1}), where 0 = 1 or 0, and 1 = number of creation"
        super().__init__(document_folder_path, name)
        group = self.name[4] # 5th letter
        if group == "0":
            self.category = 0
            self.target = Matrix([[0], [1]])
        else:
            group == "1"
            self.category = 1
            self.target = Matrix([[1], [0]])

class Observation:
        # numpy
    def get_image_info(self, image=None):
        # returns a Tensor_3D obj of Pixel
        color = True
        if image == None:
            image = self.raw_info
        assert type(image) == Image_File
        image_np_array = io.imread(image.path)
        # print(image.name)
        # print(image_np_array)
        image_array = self.image_reshape(np_array=image_np_array, color=color)
        image_np_array = np.array(image_array)

        image_tensor = Tensor_3D(image_array)
        return image_tensor
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
                    new_row.append((entry_type) (adjusted_entry))
                new_matrix.append(new_row)# = np_array[i][j][0]
            new_array.append(new_matrix)
        
        return new_array
    def __init__(self, information, target):
        self.set_raw_info(information)
        self.set_info(information)
        self.set_target(target)

    def set_raw_info(self, raw_info):
        self.raw_info = raw_info

    def set_info(self, info):
        if type(self.raw_info) == Image_File:
            info = self.get_image_info(self.raw_info)
        self.info = info

    def set_target(self, target):
        self.target = target

def collect_images():
    folderpath = IMAGE_FOLDERPATH
    observations = []
    for filename in os.listdir(folderpath):
        if filename.split(".")[-1] in ["jpeg", "jpg"]:
            image = Image_File(folderpath, filename)
            observation = Observation(image, image.target)
            observations.append(observation)
    return observations
