import numpy as np
from optparse import OptionParser
from src.networks.configmodel import init_neural_closure

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from icecream import ic


def main(legacy: bool):
    nw_width = 100
    nw_depth = 3
    spatial_dim = 2
    poly_degree = 2
    folder_name = "../dalotia_evaluation/build_new/benchmarks/NeuralClosure/Monomial_Mk11_M3_2D_gamma3/"
    # --- M1 1D synthetic tests  ----
    if options.legacy:
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
        legacy_model = True
        imported = tf.keras.models.load_model(folder_name + "best_model")
        neural_closure.model_legacy = imported
        test_model = neural_closure.model_legacy
    else:
        assert False

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


def build_new_legacy_model_for_debug_output():
    # generate a new untrained model
    nw_width = 100
    nw_depth = 3
    spatial_dim = 2
    poly_degree = 2
    folder_name = "../dalotia_evaluation/build_new/benchmarks/NeuralClosure/Monomial_Mk11_M3_2D_gamma3/"
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
    # save to new location
    model.model.save(folder_name + "best_model_")


if __name__ == "__main__":
    print("---------- Start Synthetic test Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option(
        "-l",
        "--legacy",
        dest="legacy",
        default=1,
        help="legacy mode for tf2.2 models",
        metavar="LEGACY",
    )
    (options, args) = parser.parse_args()
    options.legacy = bool(int(options.legacy))

    build_new_legacy_model_for_debug_output()
    main(legacy=options.legacy)

## currently error:
#   File "/scr/pollinta/neuralEntropyClosures/forward_pass.py", line 86, in build_new_legacy_model_for_debug_output
#     model.model.save(folder_name + "best_model_")
#   File "/scr/pollinta/neuralEntropyClosures/venv9/lib/python3.9/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
#     raise e.with_traceback(filtered_tb) from None
#   File "/scr/pollinta/neuralEntropyClosures/venv9/lib/python3.9/site-packages/keras/src/saving/legacy/saving_utils.py", line 97, in raise_model_input_error
#     raise ValueError(
# ValueError: Model <src.networks.entropymodels.SobolevModel object at 0x7faf6af7d550> cannot be saved either because the input shape is not available or because the forward pass of the model is not defined.To define a forward pass, please override `Model.call()`. To specify an input shape, either call `build(input_shape)` directly, or call the model on actual data using `Model()`, `Model.fit()`, or `Model.predict()`. If you have a custom training step, please make sure to invoke the forward pass in train step through `Model.__call__`, i.e. `model(inputs)`, as opposed to `model.call()`.
