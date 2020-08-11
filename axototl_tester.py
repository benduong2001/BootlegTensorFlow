from axototl import *

dister = DISTRIBUTION_VERIFIER()

def test_drvs():

    Am = [[[1, 1, 1],
           [0, 1, 1],
           [0, 1, 0]]]
    fm = [[[0, 1],
           [1, 1]]]
    bm = [[[0, 0],[0, 0]]]
    b = Tensor_3D(bm)
    A = Tensor_3D(Am)
    f = Tensor_3D(fm)
    fl = [f]
    bl = [b]
    cnn_l0 = Conv_Layer(network = samp_conv,
                layer_rank = 0,
                a0 = A,
                Fs = fl,
                Bs = bl,
                stride = 1,
                pad = 0)
    def test_Fs_drv():
        dn = Tensor_3D([[[100, 100], [100, 100]]])
        dr = Tensor_3D([[[300, 400], [200, 300]]])
        drp = cnn_l0.drv_Fs(chrt = dn)
        try:
            assert drp[0].unwrap() == dr.unwrap()
        except:
            print("Error for Drv")
            print(dr)
            print("^ true")
            print(drp)
            print("^ fake")
    test_Fs_drv()
    samp_conv.all_layers.clear()
    samp_conv.all_parameters.clear()
def test_configuring():
    J = 1

    NI1 = Tensor_3D([
        [[J for _ in range(5)]
         for _ in range(5)]
    ])
    t = Matrix([[1], [1]])
    
    NF1 = Tensor_3D([
        [[J for _ in range(2)]
         for _ in range(2)]
        for _ in range(3)
    ])
    NF2 = NF1.__deepcopy__()
    NB1 = Tensor_3D([
        [[J for _ in range(4)]
         for _ in range(4)],
        
        [[J for _ in range(4)]
         for _ in range(4)]
         ])
    NFL = [NF1, NF2]
    NBL = [NB1]
    pm = [NFL, NBL]
    samp_conv.reconfigure_observation(NI1, t)
    samp_conv.GD3_reconfigure_parameters(pm)
    samp_conv.GD4_update_reconfigurations()
    sp = samp_conv.all_parameters
    



    J = 0.5

    NF1 = Tensor_3D([
        [[J for _ in range(1)]
         for _ in range(1)]
        
    ])
    NF2 = NF1.__deepcopy__()
    NB1 = Tensor_3D([
        [[J for _ in range(4)]
         for _ in range(4)],
        
        [[J for _ in range(4)]
         for _ in range(4)]
         ])
    NFL = [NF1, NF2]
    NBL = [NB1]
    pm = [NFL, NBL]    
    
    samp_conv.GD3_reconfigure_parameters(pm)


def default_categ_nn(conv_a1):
    standard = Matrix([[1 + float(np.random.randn()) for _ in range(8)]
                               for _ in range(2)])
    Softboi = Categ_NN()
    l0 = Layer(layer_rank=0,
               network=Softboi,
               weights=Matrix([[1, 2, 3, 2, 1, 4, 5, 1],
                               [1, 5, 2, 3, 2, 8, 6, 7]]),
               biases=Matrix([[1], [1]]),
               inputs=conv_a1,
               activation=SOFTMAX()
               )
    a1 = l0.forward()
    l0.a1 = a1
    target = Matrix([[0], [1]])
    ll = Loss(a0=a1, network=Softboi, freq_target=target,
              loss_function=CROSSENTROPY())
    l0.next_layer = ll
    a2 = ll.forward(a1, target)
    ll.a1 = a2
    return Softboi


I1 = Tensor_3D([
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
    ])
F1 = Tensor_3D([
    [[1, 0],
     [0, 1]],
    
    [[1, 1],
     [0, 1]],

    [[0, 1],
     [0, 1]]
    
    ])
F2 = F1.__deepcopy__()
B1 = Tensor_3D([
    [[1 for _ in range(4)]
     for _ in range(4)],
    
    [[2 for _ in range(4)]
     for _ in range(4)]
     ])
FL = [F1, F2]
BL = [B1]
samp_conv = Conv_nn()

test_drvs()

cnn_l0 = Conv_Layer(network = samp_conv,
                    layer_rank = 0,
                    a0 = I1,
                    Fs = FL,
                    Bs = BL,
                    stride = 1,
                    pad = 0)
l0a1 = cnn_l0.forward()
cnn_l0.a1 = l0a1
# print(l0a1)
"""
[[[7 4 4 5]
  [5 3 5 3]
  [5 4 3 6]
  [4 6 4 3]]

 [[8 5 5 6]
  [6 4 6 4]
  [6 5 4 7]
  [5 7 5 4]]]"""
cnn_l1 = Pool_Layer(network = samp_conv,
                    layer_rank = 1,
                    a0 = l0a1,
                    stride = 2,
                    pad = 0)
cnn_l0.next_layer = cnn_l1
l1a1 = cnn_l1.forward()
cnn_l1.a1 = l1a1

# print(l1a1)
"""
[[[7 5]
  [6 6]]

 [[8 6]
  [7 7]]]"""

colvec = (Vector(l1a1.unwrap())).to_matrix()
softboi = default_categ_nn(colvec)


cnn_l2 = Fully_Connected_Layer(network = samp_conv,
                    layer_rank = 0,
                    a0 = l1a1,
                    categ_nn = softboi)
cnn_l1.next_layer = cnn_l2
l2a1 = cnn_l2.forward()
cnn_l1.a1 = l2a1

# print(cnn_l2.categ_nn.final_loss)
# ln(0.5) is around -0.693....
"""
Loss;
A0-[[0.5]
 [0.5]];
target-[[0]
 [1]];
a1-[[-0.        ]
 [-0.69314716]]"""


# print(cnn_l2.drv_a0())

conv_grad = samp_conv.GD1_create_gradient()
def conv_categ_test(Softboi):
    Softboighost = Conv_Categ_NN_Executor(Softboi, None)
    Softboighost.set_test_verifier_holder(dister)
    accuracy = Softboighost.main_executor()
    return accuracy

def categ_test_samples(Softboi):
    accuracy = conv_categ_test(Softboi)
    print(accuracy)
    return accuracy

(categ_test_samples(samp_conv))  

