import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetV2B0


def calc_euclidian_dists(x, y):
    """
    Calculate Euclidean distance between two 3D tensors.

    Args:
        x (tf.Tensor): Query embeddings (n_query, embedding_dim).
        y (tf.Tensor): Prototype embeddings (n_class, embedding_dim).

    Returns (tf.Tensor): 2D tensor with distances (n_query, n_class).
    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(
        tf.expand_dims(x, 1), [1, m, 1]
    )  # (n_query, 1, embedding_dim) -> (n_query, n_class, embedding_dim)
    y = tf.tile(
        tf.expand_dims(y, 0), [n, 1, 1]
    )  # (1, n_class, embedding_dim) -> (n_query, n_class, embedding_dim)
    return tf.reduce_sum(
        tf.square(x - y), axis=2
    )  # Return squared Euclidean distance (n_query, n_class)


class PrototypicalLayer(tf.keras.layers.Layer):
    """
    Custom Prototypical Layer that computes the prototypes and
    calculates the loss and accuracy based on Euclidean distance.
    """

    def __init__(self, name=None):
        super(PrototypicalLayer, self).__init__(name=name)

    def call(self, z_query, z_prototypes, y, n_class, n_query):
        """
        Forward pass of the Prototypical Layer to calculate loss and accuracy.

        Args:
            z_query (tf.Tensor): Query embeddings.
            z_prototypes (tf.Tensor): Prototypes computed from support embeddings.
            y (np.array): Ground truth labels for query samples.
            n_class (int): Number of classes.
            n_query (int): Number of query samples per class.

        Returns:
            loss (tf.Tensor): Calculated loss based on Euclidean distance.
            acc (tf.Tensor): Accuracy of the model.
        """

        # Calculate distances between query embeddings and prototypes
        dists = calc_euclidian_dists(z_query, z_prototypes)

        # Log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])

        # Create one-hot encoding of ground truth labels
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)

        # Calculate loss as the negative log probability
        loss = -tf.reduce_mean(tf.reduce_sum(y_onehot * log_p_y, axis=-1))

        # Calculate accuracy
        eq = tf.cast(tf.equal(tf.argmax(log_p_y, axis=-1), y), tf.float32)
        acc = tf.reduce_mean(eq)

        return loss, acc


class PrototypicalModel(Model):
    """
    Prototypical Network implementation with the encoder and the custom Prototypical Layer.
    """

    def __init__(self, n_support, n_query, w, h, c):
        """
        Args:
            n_support (int): Number of support examples.
            n_query (int): Number of query examples.
            w (int): Image width.
            h (int): Image height.
            c (int): Number of channels.
        """
        super(PrototypicalModel, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.n_support = n_support
        self.n_query = n_query

        # Use EfficientNetV2 B0 as the encoder without the top classification layer
        self.encoder = EfficientNetV2B0(
            input_shape=(w, h, c),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )

        # Initialize Prototypical Layer
        self.prototypical_layer = PrototypicalLayer()

    @tf.function  # Improves performance by compiling the graph
    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]

        # Ground truth labels for the query samples
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))

        # Merge support and query to forward through the encoder
        cat = tf.concat(
            [
                tf.reshape(support, [n_class * n_support, self.w, self.h, self.c]),
                tf.reshape(query, [n_class * n_query, self.w, self.h, self.c]),
            ],
            axis=0,
        )
        z = self.encoder(cat)

        # Divide embedding into support and query
        z_support = tf.reshape(z[: n_class * n_support], [n_class, n_support, -1])
        z_query = z[n_class * n_support :]

        # Prototypes are means of n_support examples
        z_prototypes = tf.reduce_mean(z_support, axis=1)

        # Use PrototypicalLayer to calculate loss and accuracy
        loss, acc = self.prototypical_layer(z_query, z_prototypes, y, n_class, n_query)

        return loss, acc

    def save(self, model_path):
        """
        Save encoder to a file in TensorFlow's format (.tf).

        Args:
            model_path (str): path to save the model.

        Returns: None
        """
        self.encoder.save_weights(model_path)

    def load(self, model_path):
        """
        Load encoder weights from a file.

        Args:
            model_path (str): path to the saved model.

        Returns: None
        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))  # Initialize weights
        self.encoder.load_weights(model_path)
