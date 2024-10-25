"""
Logic for model creation, training launching, and actions needed to be
accomplished during training (metrics monitoring, model saving, etc.).
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Ensure GPU memory growth for TensorFlow
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

# Import necessary components
from prototf.models import PrototypicalModel
from prototf.data import load
from prototf import TrainEngine


def train(config):
    """
    Train the Prototypical network with specified configurations.

    Args:
        config (dict): Configuration dictionary with training parameters.
    """
    # Set random seeds for reproducibility
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Create directory for saving the model
    model_dir = os.path.dirname(config["model.save_path"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load datasets (train and validation)
    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ["train", "val"])
    train_loader = ret["train"]
    val_loader = ret["val"]

    # Determine device (GPU or CPU)
    if config["data.cuda"]:
        cuda_num = config["data.gpu"]
        device_name = f"GPU:{cuda_num}"
    else:
        device_name = "CPU:0"

    # Setup model and training configuration
    n_support = config["data.train_support"]
    n_query = config["data.train_query"]
    w, h, c = list(map(int, config["model.x_dim"].split(",")))
    model = PrototypicalModel(n_support, n_query, w, h, c)

    # **Added: Model Summary**
    model.encoder.build(input_shape=(None, w, h, c))  # Build model with input shape
    model.encoder.summary()  # Display the model architecture

    optimizer = tf.keras.optimizers.Adam(config["train.lr"])

    # Define metrics to track training and validation loss/accuracy
    train_loss = tf.metrics.Mean(name="train_loss")
    val_loss = tf.metrics.Mean(name="val_loss")
    train_acc = tf.metrics.Mean(name="train_accuracy")
    val_acc = tf.metrics.Mean(name="val_accuracy")
    val_losses = []

    # Loss function for prototypical network
    @tf.function
    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    # Training step logic
    @tf.function
    def train_step(loss_func, support, query):
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_acc(acc)

    # Validation step logic
    @tf.function
    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    # Create and setup training engine with hooks for various stages
    train_engine = TrainEngine()

    # Hooks for tracking and printing status during training
    def on_start(state):
        print("Training started.")

    train_engine.hooks["on_start"] = on_start

    def on_end(state):
        print("Training ended.")

    train_engine.hooks["on_end"] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()

    train_engine.hooks["on_start_epoch"] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state["epoch"]
        template = "Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_acc.result() * 100,
                val_loss.result(),
                val_acc.result() * 100,
            )
        )

        cur_loss = val_loss.result().numpy()
        if cur_loss < state["best_val_loss"]:
            print("Saving new best model with loss:", cur_loss)
            state["best_val_loss"] = cur_loss
            model.save(config["model.save_path"])
        val_losses.append(cur_loss)

        # Early stopping logic based on validation loss
        patience = config["train.patience"]
        if len(val_losses) > patience and max(val_losses[-patience:]) == val_losses[-1]:
            state["early_stopping_triggered"] = True

    train_engine.hooks["on_end_epoch"] = on_end_epoch

    # Hook to handle episodes (batches within an epoch)
    def on_start_episode(state):
        if state["total_episode"] % 20 == 0:
            print(f"Episode {state['total_episode']}")
        support, query = state["sample"]
        loss_func = state["loss_func"]
        train_step(loss_func, support, query)

    train_engine.hooks["on_start_episode"] = on_start_episode

    # Validation at the end of each episode
    def on_end_episode(state):
        val_loader = state["val_loader"]
        loss_func = state["loss_func"]
        for i_episode in range(config["data.episodes"]):
            support, query = val_loader.get_next_episode()
            val_step(loss_func, support, query)

    train_engine.hooks["on_end_episode"] = on_end_episode

    # Train the model
    time_start = time.time()
    with tf.device(device_name):
        train_engine.train(
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["train.epochs"],
            n_episodes=config["data.episodes"],
        )
    time_end = time.time()

    # Print total training time
    elapsed = time_end - time_start
    h, min = divmod(elapsed // 60, 60)
    sec = elapsed % 60
    print(f"Training took: {h} h {min} min {sec:.2f} sec")
