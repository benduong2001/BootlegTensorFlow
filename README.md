# BootlegTensorFlow

Very basic and simplified modeler for feed-forward neural networks, built from scratch. Can support convolution neural networks, and possibly support vector machines (but cannot currently support recurrent neural network). The reason was to give a bit more under-the-hood understanding of basic neural networks without tensorflow's blackbox effect.

* You can try out BTF_mn_classification_test_run.py, which will give a sample run of a neural network learning to seperate the 3 clusters (-20< = x <= 0), (0 <= x <= 25), (26 <= x <= 80)

* The convolutional neural network's sample run involves distinguishing between 5x5 drawings of 1 or 0. 
* You must first run convnet_image_creator.py, which will generate the dataset of images in another folder called convnet_images.
* Then, you would run Convnet_drawinger which opens a tkinter 5x5 canvas where you can submit your own drawings of 1 or 0 for the model to test. When you finish drawing 1 or 0, you need to click save first, and then you can click test. The console will guess the number.
