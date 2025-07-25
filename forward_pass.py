import numpy as np
from optparse import OptionParser
from src.networks.configmodel import init_neural_closure

import tensorflow as tf
from icecream import ic
import subprocess


def main(legacy: bool, folder_name: str) -> None:
    nw_width = 100
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
        )
        neural_closure.create_model()
        ### Need to load this model as legacy code
        print("Load model in legacy mode. Model was created using tf 2.2.0")
        imported = tf.keras.models.load_model(folder_name + "best_model")
        neural_closure.model_legacy = imported
        test_model = neural_closure.model_legacy
    else:
        neural_closure = init_neural_closure(network_mk=11, poly_degree=poly_degree, spatial_dim=spatial_dim,
                                             folder_name="../" + folder_name,
                                             loss_combination=2, nw_width=nw_width, nw_depth=nw_depth,
                                             normalized=True, input_decorrelation=True,
                                             scale_active=True)
        neural_closure.load_model()
        test_model = neural_closure.model

    # read binary float32 input data and convert to tensor
    # load data
    kitrt_servingSize = 12920
    num_input_channels = 9
    u = np.fromfile(
        "../dalotia_evaluation/benchmarks/NeuralClosure/inputs1367.bin",
        dtype=np.float32,
    )
    u = u.reshape((kitrt_servingSize, num_input_channels))
    input_tensor = tf.convert_to_tensor(u, dtype=tf.float32)

    _, alpha, _ = ic(test_model(input_tensor))
    assert alpha.shape == (kitrt_servingSize, num_input_channels), (
        "Output shape mismatch"
    )
    assert alpha[0][0] == -0.6211774349212646, (
        "First element should be -0.6211774349212646, but is {}".format(alpha[0][0])
    )


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
    folder_name = "../dalotia_evaluation/build_new/benchmarks/NeuralClosure/Monomial_Mk11_M3_2D_gamma3/"

    build_new_legacy_model_for_debug_output(folder_name)
    ic("moving model files to best_model")
    # move model files from the best_model_ to best_model ; omit variables folder
    subprocess.run(
        ["mv", folder_name + "best_model_/saved_model.pb", folder_name + "best_model/saved_model.pb"],
        check=True,
    )
    subprocess.run(
        ["mv", folder_name + "best_model_/keras_metadata.pb", folder_name + "best_model/keras_metadata.pb"],
        check=True,
    )
    ic("trying to load model")
    main(legacy=False, folder_name=folder_name)
