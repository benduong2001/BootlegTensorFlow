# BootlegTensorFlow

Very basic and simplified modeler for feed-forward neural networks, built from scratch. Can support convolution neural networks, and possibly support vector machines (but cannot currently support recurrent neural network). The reason was to give a bit more under-the-hood understanding of basic neural networks without its usual blackbox effect, down to the backpropagation of the matrices.

* The convolutional neural network's example run involves distinguishing between 5x5 drawings of 1 or 0, but these images can be trained.
* You must first run convnet_image_creator.py, which will generate the dataset of images in another folder called convnet_images.
* Then, you would run Convnet_drawinger which opens a tkinter 5x5 canvas where you can submit your own drawings of 1 or 0 for the model to test. When you finish drawing 1 or 0, you need to click save first, and then you can click test. The console will guess the number.

![](bootlegCNN_demo_gif.gif)
