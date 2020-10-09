# Sample use of matrixed_network 10_4_2020 on
# linear regression

from matrixed_network_10_4_2020 import * 

def linreg_test():
    lrn = Categ_NN()
    l0 = Layer(layer_rank=0,
               network=lrn,
               weights=Matrix([[1, 3],
                               [2, 4]]),
               biases=Matrix([[1], [1]]),
               inputs=Matrix([[1], [2]]),
               activation=SIGMOID()
               )
    a1 = l0.forward()

    l1 = Layer(layer_rank=1,
               network=lrn,
               weights=Matrix([[1, 2]]),
               biases=Matrix([[1]]),
               inputs=a1,
               activation=SIGMOID()
               )
    l0.a1 = a1
    l0.next_layer = l1
    a2 = l1.forward()
    target = Matrix([[0]])
    ll = Loss(a0=a2, network=lrn, freq_target=target,
              loss_function=MEAN_SQUARED_ERROR())
    a3 = ll.forward(a2, target)
    l1.a1 = a2
    l1.next_layer = ll
    ll.a1 = a3
    lrne = Categ_NN_Executor(lrn, LINREG_ENCODER())
    lrne.set_test_verifier_holder(LINREG_VERIFIER())
    accuracy = lrne.main_executor()
    return accuracy


linreg_test()
