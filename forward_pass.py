import numpy as np
from src.networks.configmodel import init_neural_closure

import tensorflow as tf
from icecream import ic


def main(legacy: bool, folder_name: str) -> None:
    nw_width = 300
    nw_depth = 3
    spatial_dim = 2
    poly_degree = 3
    if legacy:
        # load network
        neural_closure = init_neural_closure(
            network_mk=11,
            poly_degree=poly_degree,
            spatial_dim=spatial_dim,
            folder_name="unused",
            loss_combination=2,
            nw_width=nw_width,
            nw_depth=nw_depth,
            normalized=True,
            input_decorrelation=True,
            basis="spherical_harmonics",
            scale_active=False,
            gamma_lvl=2,
            rotated=False,
        )
        neural_closure.create_model()
        ### Need to load this model as legacy code
        print("Load model in legacy mode. Model was created using tf 2.2.0")
        imported = tf.keras.models.load_model(folder_name + "best_model")
        neural_closure.model_legacy = imported
        test_model = neural_closure.model_legacy
        # write again to file
        test_model.save("./best_model")
    else:
        neural_closure = init_neural_closure(
            network_mk=11,
            poly_degree=poly_degree,
            spatial_dim=spatial_dim,
            folder_name="../" + folder_name,
            loss_combination=2,
            nw_width=nw_width,
            nw_depth=nw_depth,
            normalized=True,
            input_decorrelation=True,
            basis="spherical_harmonics",
            scale_active=False,
            gamma_lvl=2,
            rotated=False,
        )
        neural_closure.load_model()
        test_model = neural_closure.model

    test_model.summary()

    # read binary float32 input data and convert to tensor
    kitrt_servingSize = 12920
    num_input_channels = 9
    u = np.fromfile(
        "../dalotia_evaluation/benchmarks/NeuralClosure/inputs1367.bin",
        dtype=np.float32,
    )
    u = u.reshape((kitrt_servingSize, num_input_channels))
    input_tensor = tf.convert_to_tensor(u, dtype=tf.float32)

    _, alpha, _ = ic(test_model(input_tensor))
    assert alpha.shape == (
        kitrt_servingSize,
        num_input_channels,
    ), "Output shape mismatch"
    assert (
        alpha[0][0] == -1.3182921409606934
    ), "First element should be -0.6211774349212646, but is {}".format(alpha[0][0])
    test_model.core_model.summary()


def build_new_legacy_model_for_debug_output(folder_name: str) -> None:
    # generate a new untrained model
    nw_width = 100
    nw_depth = 3
    spatial_dim = 2
    poly_degree = 3
    model = init_neural_closure(
        network_mk=11,
        poly_degree=poly_degree,
        spatial_dim=spatial_dim,
        folder_name="unused",
        loss_combination=2,
        nw_width=nw_width,
        nw_depth=nw_depth,
        normalized=True,
    )
    model.create_model()
    kitrt_servingSize = 12920
    num_input_channels = 9
    training_data = np.zeros((kitrt_servingSize, num_input_channels), dtype=np.float32)
    model.model(training_data)
    # save to new location
    model.model.save(folder_name + "best_model_")


if __name__ == "__main__":
    folder_name = "../dalotia_evaluation/build_new/benchmarks/NeuralClosure/Harmonic_Mk11_M3_2D_gamma2/"
    ic("trying to save legacy model")
    main(legacy=True, folder_name=folder_name)

    folder_name = "./"
    ic("trying to load non-legacy model")
    main(legacy=False, folder_name=folder_name)
