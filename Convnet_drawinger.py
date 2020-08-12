from tkinter import *
import os
import numpy as np
import tkinter as tk
from tkinter import Button, Canvas, Pack
from PIL import Image
from axototl_tester import *
def test_convtemp_image(samp_conv):
    # tests the image just created as conv% (temp) by convnet drawinger
    folderpath = "C:/Users/Benson/Desktop/BootlegTensorFlowFolder"
    for filename in os.listdir(folderpath):
        if filename.split(".")[-1] in ["jpeg", "jpg"] and "temp" in filename:
            # there is only one image file, conv% temp
            image = Image_File(folderpath, filename)
            temp_observation = Observation(image, image.target)
    temp_input_ = temp_observation.info
    temp_target = temp_observation.target
    samp_conv.reconfigure_observation(temp_input_, temp_target)
    samp_conv.GD4_update_reconfigurations()

    pred_column = samp_conv.all_layers[-1].categ_nn.final_loss.a0
    pred_dist = [abs(x) for x in pred_column.unwrap()]
    pred_max_idx = max(range(len(pred_dist)), key=pred_dist.__getitem__)

    if pred_max_idx == 0:
        pred_label = 1
    elif pred_max_idx == 1:
        pred_label = 0
    return pred_label

black_max = "#282828"
white_min = "#d7d7d7"

ink = "#282828";
factor = 100
compartments = 5
main_width = 5 * factor
main_height = 5 * factor
brush_size = 1 * factor

class Canvas_Pixel:
    def __init__(self, hex_, corners):
        self.hex_ = hex_
        self.corners = corners # [x0, y0, x1, y1]
    def __repr__(self):
        return str(self.corners) + ", " +  str(self.hex_)

def set_up_present_labels():
    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments

    present_labels = []
    for y in range(0, compartments):
        row = []
        for x in range(0, compartments):
            corners = (x * (compartment_width),
                       y * (compartment_height),
                       x * (compartment_width) + (compartment_width) - 1,
                       y * (compartment_height) + (compartment_height) -1)
            print(corners)
            canvas_pixel = Canvas_Pixel("#ffffff", corners)
            row.append(canvas_pixel)
        present_labels.append(row)
    assert len(present_labels) == 5
    assert len(present_labels[0]) == 5
    return present_labels
present_labels = set_up_present_labels()

categ = "1"
##def categ1():
##    global categ 
##    categ = "1" 
##def categ0():
##    global categ
##    categ = "0"

def start_paint(event):
    canvas.bind('<B1-Motion>', painter)
def start_erase(event):
    canvas.bind('<B3-Motion>', eraser)
    lastx, lasty = event.x, event.y


HEX_CONVERTER = list("0123456789abcdef")
len_HEX_CONVERTER = len(HEX_CONVERTER)

def hexify_channel(channel_hex):
    assert len(channel_hex) == 2
    channel_hex = channel_hex.lower()
    part1,part2 = list(channel_hex)
    channel_part1 = HEX_CONVERTER.index(part1)
    channel_part2 = HEX_CONVERTER.index(part2)
    value = (channel_part1 * len_HEX_CONVERTER) + channel_part2 
    return value
def hex_to_rgb(hex_str):
    # remove pound symnbol
    hex_str = hex_str[1:]
    assert len(hex_str) == 6
    rgb_values = []
    for i in range(0, 6, 2):
        channel_hex = hex_str[i:i + 2]
        channel_value = hexify_channel(channel_hex)
        rgb_values.append(channel_value)
    return rgb_values
def rgb_to_hex(rgb):
    # list of ints from 0 to 255
    hex_str = "#"
    for channel_value in rgb:
        part1 = channel_value // len_HEX_CONVERTER
        part2 = channel_value % len_HEX_CONVERTER
        channel_part1 = HEX_CONVERTER[part1]
        channel_part2 = HEX_CONVERTER[part2]
        hex_str = hex_str + channel_part1 + channel_part2
    return hex_str

def color_opacity(color_hex_str, direction = True):
    # returns color as HEX_STR
    # black = false, white = true
    # black = -10; white +10
    f = 10 # factor of adjustment for increase/decrease
    rgb_form = hex_to_rgb(color_hex_str)
    r, g, b = rgb_form
    if direction == True:
        new_rgb_form = [min([255, r + f]),
                        min([255, g + f]),
                        min([255, b + f])]
        # increases the channels, but ceilinged at 255
    else:
        new_rgb_form = [max([0, r - f]),
                        max([0, g - f]),
                        max([0, b - f])]
    new_color_hex_str = rgb_to_hex(new_rgb_form)
    # print("new hex {0}".format(new_color_hex_str))
    return new_color_hex_str

def color_update(corners, direction):
    # corner, tuple of 4 ints from 0 to canvas length
    # outputs new color, a hex_string
    


    
    curr_canvas_pixel = present_labels[corners[1]//100][corners[0]//100]
    old_color = curr_canvas_pixel.hex_
    old_rgb = hex_to_rgb(old_color)
    # deciding new color: if same direction, color_opacity,
    # if other direction, new edge color
    old_direction = float(old_rgb[0]) > 127.5 # true if above, false if below
    # print(old_rgb[0])
    if direction == old_direction:
        # print("new direct")
        new_color = color_opacity(old_color, direction)
    else:
        # diverge
        if direction == True: # if diverging to white
            new_color = white_min
        else: # if diverging to black
            new_color = black_max
    present_labels[corners[1]//100][corners[0]//100] = Canvas_Pixel(new_color, corners)
    return new_color

def floored_corners_paint(eventx, eventy):

    
    global brush_size, compartments, main_width, main_height

    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments

    event_x = max([0, min([main_width - 1, eventx])]) # to prevent going off camera
    event_y = max([0, min([main_height - 1, eventy])])

    assert event_y >= 0 and event_y < 500
    assert event_x >= 0 and event_x < 500
    
    anchor_x0 = ((event_x // brush_size) * compartment_width)
    anchor_y0 = ((event_y // brush_size) * compartment_height)
    anchor_x1 = ((event_x // brush_size) * compartment_width) + compartment_width
    anchor_y1 = ((event_y // brush_size) * compartment_height) + compartment_height
    
    
    corners = (anchor_x0, anchor_y0, anchor_x1 - 1, anchor_y1 - 1)
    return corners


def painter(event):
    # global lastx, lasty
    corners = floored_corners_paint(event.x, event.y)

    # deciding new color: if same side, color_opacity, else, 
    new_color = color_update(corners, False)
    # print("newcolor is")
    # print(new_color)
    canvas.create_rectangle(corners, fill=new_color, width=1, outline=new_color)
    # mini_corners = mini_paint(corners)
    # canvas2.create_rectangle(mini_corners, fill='white', width=1, outline="white")

def eraser(event):
    corners = floored_corners_paint(event.x, event.y)
    new_color = color_update(corners, True)
    canvas.create_rectangle(corners, fill=new_color, width=1, outline=new_color)
    # mini_corners = mini_paint(corners)
    # canvas2.create_rectangle(mini_corners, fill='white', width=1, outline="white")

def tester():
    global samp_conv
    answer = test_convtemp_image(samp_conv)
    print("Is this a {0}, human?".format(str(answer)))


root = tk.Tk()

def save():
    global categ
    filename = "conv" + categ + " (temp)" + ".jpg"

    compartment_width = main_width // compartments # aka 100, if main_width = 500 and compartments = 5
    compartment_height = main_height // compartments
    
    image_matrix = []
    for y in range(0, compartments):
        image_row = []
        for x in range(0, compartments):
            canvas_pixel = present_labels[y][x]
            image_row.append(list(hex_to_rgb(canvas_pixel.hex_)))
        image_matrix.append(image_row)
    image_array = np.array(image_matrix)
    assert image_array.shape == (5, 5, 3)
    image = Image.fromarray(image_array.astype("uint8"))
    folderpath = "C:/Users/Benson/Desktop/BootlegTensorFlowFolder"
    for oldfilename in os.listdir(folderpath):
        if "conv1 (temp)" in oldfilename:
            os.remove(oldfilename)
        elif "conv0 (temp)" in oldfilename:
            os.remove(oldfilename)
    image.save(filename)


# button1 = Button(text = "1", command = categ1)
# button1.pack()
# button0 = Button(text = "0", command = categ0)
# button0.pack()
button_tester = Button(text = "Test", command = tester)
button_tester.pack()



canvas = Canvas(root, width = main_width, height = main_height, bg='white')

# canvas is the zoomed in canvas for clarity, but we will only save the smol canvas, canvas2
canvas2 = Canvas(root, width = 50, height = 50, bg='red')

canvas.bind('<1>', start_paint)
canvas.bind('<3>', start_erase)
canvas.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack()

root.mainloop()
