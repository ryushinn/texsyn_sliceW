import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, List
from equinox import field
import numpy as np


class VGG19(eqx.Module):
    block1: list
    block2: list
    block3: list
    block4: list
    activation: Callable = field(static=True)
    downsampling: Callable = field(static=True)

    def __init__(self, key):
        keys = jax.random.split(key, 12)
        self.block1 = [
            eqx.nn.Conv2d(
                3, 64, 3, key=keys[0], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                64, 64, 3, key=keys[1], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block2 = [
            eqx.nn.Conv2d(
                64, 128, 3, key=keys[2], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                128, 128, 3, key=keys[3], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block3 = [
            eqx.nn.Conv2d(
                128, 256, 3, key=keys[4], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[5], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[6], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                256, 256, 3, key=keys[7], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.block4 = [
            eqx.nn.Conv2d(
                256, 512, 3, key=keys[8], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[9], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[10], padding="SAME", padding_mode="REFLECT"
            ),
            eqx.nn.Conv2d(
                512, 512, 3, key=keys[11], padding="SAME", padding_mode="REFLECT"
            ),
        ]

        self.activation = jax.nn.relu
        self.downsampling = eqx.nn.AvgPool2d((2, 2), stride=2)

    def __call__(self, x):
        features = []

        x = x[[2, 1, 0], ...]

        x = 255 * x - jnp.array([103.939, 116.779, 123.68]).reshape(3, 1, 1)

        # block1
        for conv in self.block1:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block2
        for conv in self.block2:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block3
        for conv in self.block3:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        # block4
        for conv in self.block4:
            x = self.activation(conv(x))
        features.append(x)

        x = self.downsampling(x)

        return features


def load_pretrained_VGG19_from_pth(pth_path):
    # get treedef from a dummy VGG
    VGG_dummy = VGG19(jax.random.key(0))
    _, treedef = jax.tree_util.tree_flatten(VGG_dummy)

    # formulate pretrained weights as corresponding leaves
    vgg_jnp = np.load(pth_path, allow_pickle=True).item()
    leaves_order = [
        "block1_conv1.weight",
        "block1_conv1.bias",
        "block1_conv2.weight",
        "block1_conv2.bias",
        "block2_conv1.weight",
        "block2_conv1.bias",
        "block2_conv2.weight",
        "block2_conv2.bias",
        "block3_conv1.weight",
        "block3_conv1.bias",
        "block3_conv2.weight",
        "block3_conv2.bias",
        "block3_conv3.weight",
        "block3_conv3.bias",
        "block3_conv4.weight",
        "block3_conv4.bias",
        "block4_conv1.weight",
        "block4_conv1.bias",
        "block4_conv2.weight",
        "block4_conv2.bias",
        "block4_conv3.weight",
        "block4_conv3.bias",
        "block4_conv4.weight",
        "block4_conv4.bias",
    ]
    leaves, _ = jax.tree_util.tree_flatten([vgg_jnp[k] for k in leaves_order])

    # unflatten back to model
    return jax.tree_util.tree_unflatten(treedef, leaves)


def create_slice_loss(features, exemplar):
    features_exemplar = features(exemplar)

    def slice_loss(sample, key):
        features_sample = features(sample)
        keys = list(jax.random.split(key, num=len(features_sample)))
        return sum(
            jax.tree_map(
                sliced_wasserstein_loss, features_exemplar, features_sample, keys
            )
        )

    return slice_loss


def sliced_wasserstein_loss(fe, fs, key):
    fe = fe.reshape(fe.shape[0], -1)
    fs = fs.reshape(fs.shape[0], -1)

    # get c random directions
    c, n = fs.shape
    Vs = jax.random.normal(key, (c, c))
    Vs = Vs / jnp.sqrt(jnp.sum(Vs**2, axis=1, keepdims=True))

    # project
    pfe = jnp.einsum("cn,mc->mn", fe, Vs)
    pfs = jnp.einsum("cn,mc->mn", fs, Vs)

    # sort
    spfe = jnp.sort(pfe, axis=1)
    spfs = jnp.sort(pfs, axis=1)
    ## apply interpolation like an image to match the dimension
    spfe = jax.image.resize(spfe, spfs.shape, method="nearest")

    # MSE
    loss = jnp.mean((spfe - spfs) ** 2)

    return loss


def create_gram_loss(features, exemplar):
    features_exemplar = features(exemplar)
    gmatrices_exemplar = jax.tree_map(gram_matrix, features_exemplar)
    mse = lambda x, y: jnp.mean((x - y) ** 2)

    def gram_loss(sample, key=None):
        # key is never used here but declared for API compatibility
        features_sample = features(sample)
        gmatrices_sample = jax.tree_map(gram_matrix, features_sample)

        loss = sum(jax.tree_map(mse, gmatrices_exemplar, gmatrices_sample))
        return loss

    return gram_loss

def gram_matrix(f):
    f = f.reshape(f.shape[0], -1)
    gram_matrix = f @ f.transpose()

    gram_matrix = gram_matrix / f.shape[-1]
    return gram_matrix

def gram_loss(features, exemplar, sample, key=None):
    features_exemplar = features(exemplar)
    gmatrices_exemplar = jax.tree_map(gram_matrix, features_exemplar)

    features_sample = features(sample)
    gmatrices_sample = jax.tree_map(gram_matrix, features_sample)

    mse = lambda x, y: jnp.mean((x - y) ** 2)
    loss = sum(jax.tree_map(mse, gmatrices_exemplar, gmatrices_sample))
    return loss