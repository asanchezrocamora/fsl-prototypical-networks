"""
Logic for evaluation procedure of saved model.
"""

import os
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Error setting memory growth")
        print(e)

sys.path.append(os.path.dirname(os.path.realpath("__file__")))
print(os.path.dirname(os.path.realpath("__file__")))

from prototf.models import PrototypicalModel
from prototf.data import load


def eval(config):
    n_support = config["data.test_support"]
    n_query = config["data.test_query"]
    (
        w,
        h,
        c,
    ) = list(map(int, config["model.x_dim"].split(",")))
    model = PrototypicalModel(n_support, n_query, w, h, c)
    model_path = f"{config['model.save_path']}"
    model.load(model_path)
    print("Model loaded.")

    # Determine device
    if config["data.cuda"]:
        cuda_num = config["data.gpu"]
        device_name = f"GPU:{cuda_num}"
    else:
        device_name = "CPU:0"

    # Load data from disk
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ["test"])
    test_loader = ret["test"]

    # Metrics to gather
    test_loss = tf.metrics.Mean(name="test_loss")
    test_acc = tf.metrics.Mean(name="test_accuracy")

    def calc_loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    with tf.device(device_name):
        for i_episode in range(config["data.episodes"]):
            support, query = test_loader.get_next_episode()
            if (i_episode + 1) % 50 == 0:
                print("Episode: ", i_episode + 1)
            loss, acc = calc_loss(support, query)
            test_loss(loss)
            test_acc(acc)

    print("Loss: ", test_loss.result().numpy())
    print("Accuracy: ", test_acc.result().numpy())


if __name__ == "__main__":
    pass
