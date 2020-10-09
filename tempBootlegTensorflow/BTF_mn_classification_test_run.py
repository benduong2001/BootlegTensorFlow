# Sample use of matrixed_network 10_4_2020 on
# general classification

from matrixed_network_10_4_2020 import * 

def categ_test():
    gn = Categ_NN()
    l0 = Layer(layer_rank=0,
               network=gn,
               weights=Matrix([[1, 1],
                               [1, 1],
                               [1, 1],
                               [1, 1],
                               [1, 1],
                               [1, 1],
                               [1, 1],
                               [1, 1]]),
               biases=Matrix([[1] for _ in range(8)]),
               inputs=Matrix([[1], [1]]),
               activation=SIGMOID()
               )
    a1 = l0.forward()
    l0.a1 = a1



    l1 = Layer(layer_rank=1,
               network=gn,
               weights=Matrix([[1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1]]),
               biases=Matrix([[1], [1], [1]]),
               inputs=a1,
               activation=SOFTMAX()
               )
    l0.next_layer = l1
    a2 = l1.forward()
    l1.a1 = a2

    target = Matrix([[0], [1], [0]])
    ll = Loss(a0=a2, network=gn, freq_target=target,
              loss_function=CROSSENTROPY())
    l1.next_layer = ll
    a3 = ll.forward(a2, target)
    ll.a1 = a3

    region_encoder_obj = REGION_ENCODER()

    gne = Categ_NN_Executor(gn, region_encoder_obj)
    gne.set_test_verifier_holder(DISTRIBUTION_VERIFIER())
    accuracy = gne.main_executor()
    return accuracy

def categ_test_samples(k = 1):
    accuracies = []
    for i in range(k):
        accuracy = categ_test()
        accuracies.append(accuracy)
    return accuracies

print(categ_test_samples())
