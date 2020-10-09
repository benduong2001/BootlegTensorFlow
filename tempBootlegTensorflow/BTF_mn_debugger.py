# debugger for BTF mn
# or Bootleg TensorFlow Matrixed Network (10_4_2020)
from matrixed_network_10_4_2020 import * 


# DEBUGGING TESTS:
def flood_resevoirs():
    def samp_categ():
        Union = Categ_NN()
        l0 = Layer(layer_rank=0,
                   network=Union,
                   weights=Matrix([[1, 3, 5],
                                   [2, 4, 6]]),
                   biases=Matrix([[1], [1]]),
                   inputs=Matrix([[1], [2], [3]]),
                   activation=SIGMOID()
                   )
        a1 = l0.forward()
        l0.a1 = a1

        l1 = Layer(layer_rank=1,
                   network=Union,
                   weights=Matrix([[1, 2]]),
                   biases=Matrix([[1]]),
                   inputs=a1,
                   activation=SIGMOID()
                   )
        l0.next_layer = l1
        a2 = l1.forward()

        target = Matrix([[0]])
        ll = Loss(a0=a2, network=Union, freq_target=target,
                  loss_function=MEAN_SQUARED_ERROR())
        a3 = ll.forward(a2, target)
        l1.a1 = a2
        l1.next_layer = ll
        ll.a1 = a3;
        return Union

    samp = samp_categ()

    def assert_equals(a, b):
        try:
            assert a == b;
            return True
        except:
            print("NOT EQUAL ERROR")
            print(a, b)
            raise AssertionError("Not equal")

    def assert_not_equals(a, b):
        try:
            assert a != b;
            return True
        except:
            print("DOES EQUAL ERROR")
            print(a, b)
            raise AssertionError("Does equal")

    def flood_resevoirs1(samp):
        l0, l1 = samp.all_layers
        ll = samp.final_loss

        assert "0.98201" in str(l1.a1.unwrap()[0])  # should be close to 0.98201

        assert "0.96435" in str(ll.a1.unwrap()[0])
        print("all resovoirs flooded")
        return True

    flood_resevoirs1(samp)

    def flood_resevoir2(samp):
        wz1 = Matrix([[0, 0, 0],
                      [0, 0, 0]])
        bz1 = Matrix([[0], [0]])
        wz2 = Matrix([[0, 0]])
        bz2 = Matrix([[0]])
        fake_parameters = [
            wz1, bz1, wz2, bz2
        ]
        samp.GD3_reconfigure_parameters(fake_parameters)
        assert_equals(samp.all_layers[0].W, wz1)
        assert_equals(samp.all_layers[0].b, bz1)
        assert_equals(samp.all_layers[1].W, wz2)
        assert_equals(samp.all_layers[1].b, bz2)

        zinp = Matrix([[0], [0], [0]])
        ztarg = Matrix([[0]])

        samp.reconfigure_observation(zinp, ztarg)
        assert_equals(samp.all_layers[0].a0, zinp)
        assert_equals(samp.final_loss.freq_target, ztarg)

        uneq = ((samp.final_loss.a1 == Matrix([[0]])))  # actually is like 0.96435108382104
        assert not uneq
        samp.GD4_update_reconfigurations()
        assert (samp.final_loss.a1 == Matrix([[0.25]]))

        wf1 = Matrix([[1, 1, 1],
                      [1, 1, 1]])
        bf1 = Matrix([[1], [1]])
        wf2 = Matrix([[1, 1]])
        bf2 = Matrix([[1]])
        fake_parameters2 = [wf1, bf1, wf2, bf2]
        samp.GD3_reconfigure_parameters(fake_parameters2)
        finp = Matrix([[1], [1], [1]])
        ftarg = Matrix([[0]])
        samp.reconfigure_observation(finp, ftarg)
        samp.GD4_update_reconfigurations()
        assert_equals(samp.all_layers[0].a1.unwrap()[0], 0.9820137900379085)
        assert_equals(samp.final_loss.a0.unwrap()[0], 0.950922299104762)  # 2.964027580075817
        assert_equals(samp.final_loss.a1.unwrap()[0], 0.9042532189346865)  # 3.8586751805229484)
        grad = samp.GD1_create_gradient()

        loss_a0_z = 2.964027580075817
        drver = SIGMOID()

        drv_loss_a0_z = drver.apply_drv(Matrix([[loss_a0_z]]))  # aka (s(x))* (1-s(x)) where s = sigmoid, x is 2.964...
        drv_final_loss = (0.950922299104762 * 2)
        assert_equals(samp.final_loss.drv_loss()[0][0],
                      drv_final_loss)
        assert (str(drv_loss_a0_z * drv_final_loss)[0][0])[:-1] in str(grad[3][0][0])  # bias

        a1_base = 0.9820137900379085
        assert_equals(grad[2][0][0], a1_base * grad[3][0][0])  # weight
        drv_greb = drver.apply_drv(Matrix([[4]]))
        # print(drv_greb) # 0.0176627062133
        # print(grad[3][0][0]) # 0.0887573380248654
        # multiplying these makes     # 0.0015676947858069693
        assert (str(drv_greb * grad[3][0][0])[0][0])[:-1] in str(grad[1][0][0])  # bias

        assert (grad[0][0][0] == finp[0][0] * grad[1][0][0])

        new_parameters = samp.GD2_finalise_gradient(fake_parameters2)
        samp.GD3_reconfigure_parameters(new_parameters)

        fv = 0.9
        nw1, nb1, nw2, nb2 = samp.all_parameters
        assert_equals(nw1, Matrix([[fv, fv, fv],
                                   [fv, fv, fv]]))
        assert_equals(nb1, Matrix([[fv], [fv]]))
        assert_equals(nw2, Matrix([[fv, fv]]))
        assert_equals(nb2, Matrix([[fv]]))

        print("all updates passed")

    flood_resevoir2(samp)

    def flood_resevoir3(samp):
        wf1 = Matrix([[1, 1, 1],
                      [1, 1, 1]])
        bf1 = Matrix([[1], [1]])
        wf2 = Matrix([[1, 1]])
        bf2 = Matrix([[1]])
        fake_parameters2 = [wf1, bf1, wf2, bf2]
        samp.GD3_reconfigure_parameters(fake_parameters2)
        finp = Matrix([[1], [1], [1]])
        ftarg = Matrix([[0]])
        samp.reconfigure_observation(finp, ftarg)
        samp.GD4_update_reconfigurations()

        obs1_inp = Matrix([[1], [1], [1]])
        target = Matrix([[0]])
        obs1 = Observation(obs1_inp, target)
        obs1.set_info(obs1.raw_info)
        obs2_inp = Matrix([[1], [1], [1]])
        obs2 = Observation(obs2_inp, target)
        obs2.set_info(obs2.raw_info)

        old_Parameters = [
            Matrix([[0.00156769, 0.00156769, 0.00156769],
                    [0.00156769, 0.00156769, 0.00156769]]),
            Matrix([[0.00156769],
                    [0.00156769]]),
            Matrix([[0.08716093, 0.08716093]]),
            Matrix([[0.08875734]])]
        avg_params = samp._average_gradient(batch_observations=[obs1, obs2],
                                            imported_parameters=samp.all_parameters)
        for i in range(len(avg_params)):
            avg_part_drv = avg_params[i]
            old_param = old_Parameters[i]
            for j in range(len(avg_part_drv.unwrap())):
                avg_part_drv_entry = avg_part_drv.unwrap()[j]
                old_param_entry = old_param.unwrap()[j]
                assert str(old_param_entry)[:-1] in str(avg_part_drv_entry)
        print("avg gradient resovoirs flooded")

    flood_resevoir3(samp)
flood_resevoirs()
