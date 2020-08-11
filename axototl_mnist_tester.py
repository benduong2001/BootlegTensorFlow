from axototl import *
from keras.datasets import mnist


dister = DISTRIBUTION_VERIFIER()



def Mnist_categ_nn(conv_a1):
    standard = Matrix([[1 + float(np.random.randn()) for _ in range(8)]
                               for _ in range(2)])
    Softboi = Categ_NN()
    l0 = Layer(layer_rank=0,
               network=Softboi,
               weights=Matrix([[1 for _ in range(250)]
                                for _ in range(50)]),
               biases=Matrix([[1] for _ in range(50)]),
               inputs=conv_a1,
               activation=SIGMOID()
               )
    a1 = l0.forward()
    l0.a1 = a1

    #"""
    l1 = Layer(layer_rank=1,
               network=Softboi,
               weights=Matrix([[1 for _ in range(50)]
                                for _ in range(10)]),
               biases=Matrix([[1] for _ in range(10)]),
               inputs=a1,
               activation=SIGMOID()
               )
    l0.next_layer = l1
    a2 = l1.forward()
    l1.a1 = a2

    samp_ohe = [[0] for _ in range(9)] + [[1]]
    target = Matrix(samp_ohe)
    l2 = Loss(a0=a2, network=Softboi, freq_target=target,
              loss_function=CROSSENTROPY())
    l1.next_layer = l2
    a3 = l2.forward(a2, target)
    l2.a1 = a3
    return Softboi #"""

I1 = Tensor_3D([[[1 for _ in range(28)]
                for _ in range(28)]])
FL = [
    Tensor_3D([
    [[1 for _ in range(3)]
      for _ in range(3)]])
      for _ in range(5)]

BL = [
    Tensor_3D([
    [[1 for _ in range(26)]
      for _ in range(26)]
      for _ in range(5)])
    ]

samp_conv = Conv_nn()

cnn_l0 = Conv_Layer(network = samp_conv,
                    layer_rank = 0,
                    a0 = I1,
                    Fs = FL,
                    Bs = BL,
                    stride = 1,
                    pad = 0)
l0a1 = cnn_l0.forward()
cnn_l0.a1 = l0a1

# a0 = 28*28*1
# F is 3*3*1, 32 filts
# B, R and a1 is 26*26*32
assert l0a1.height == 26
assert l0a1.width == 26
assert l0a1.depth == 5

cnn_l1 = Pool_Layer(network = samp_conv,
                    layer_rank = 1,
                    a0 = l0a1,
                    stride = 2,
                    pad = 0)
cnn_l0.next_layer = cnn_l1
l1a1 = cnn_l1.forward()
cnn_l1.a1 = l1a1

# poolstride is 2
# a1 is 13*13*32
###############
assert l1a1.height == 13
assert l1a1.width == 13
assert l1a1.depth == 5

FL2 = [
    Tensor_3D([
    [[1 for _ in range(3)]
      for _ in range(3)]
    for _ in range(5)
    ])
    
      for _ in range(10)]
BL2 = [
    Tensor_3D([
    [[1 for _ in range(11)]
      for _ in range(11)]
      for _ in range(10)])
    ]

cnn_l2 = Conv_Layer(network = samp_conv,
                    layer_rank = 0,
                    a0 = l1a1,
                    Fs = FL2,
                    Bs = BL2,
                    stride = 1,
                    pad = 0)
cnn_l1.next_layer = cnn_l2
l2a1 = cnn_l2.forward()
cnn_l2.a1 = l2a1

# a0 = 26*26*32
# F is 3*3*32, 64 filts
# B, R and a1 is 11*11*64
print(l2a1.height, l2a1.width, l2a1.depth)

assert l2a1.height == 11
assert l2a1.width == 11
assert l2a1.depth == 10

cnn_l3 = Pool_Layer(network = samp_conv,
                    layer_rank = 1,
                    a0 = l2a1,
                    stride = 2,
                    pad = 0)
cnn_l2.next_layer = cnn_l3
l3a1 = cnn_l3.forward()
cnn_l3.a1 = l3a1

# poolstride is 2
# a1 is 5*5*64
print(l3a1.height, l3a1.width, l3a1.depth)
assert l3a1.height == 5
assert l3a1.width == 5
assert l3a1.depth == 10

colvec = (Vector(l1a1.unwrap())).to_matrix()
softboi = Mnist_categ_nn(colvec)

print( l3a1.height, l3a1.width, l3a1.depth )#== 1600

cnn_l4 = Fully_Connected_Layer(network = samp_conv,
                    layer_rank = 0,
                    a0 = l3a1,
                    categ_nn = softboi)
cnn_l3.next_layer = cnn_l4
l4a1 = cnn_l4.forward()
cnn_l4.a1 = l4a1



def conv_categ_mnist_test(Softboi):
    global mnist
    Softboighost = Conv_Categ_NN_Executor(Softboi, None)
    Softboighost.set_test_verifier_holder(dister)
    accuracy = Softboighost.MNIST_executor(mnist)
    return accuracy

def categ_test_samples(Softboi):
    accuracies = []
    for i in range(1):
        accuracy = conv_categ_mnist_test(Softboi)
        accuracies.append(accuracy)
    return accuracies

print(categ_test_samples(samp_conv))  




