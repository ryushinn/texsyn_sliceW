import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import jax, optax, equinox as eqx
import jax.numpy as jnp
import metrics

from utils import preprocess_exemplar


parser = argparse.ArgumentParser("Pixel-wise optimization for texture synthesis")

parser.add_argument("--exemplar_path", type=str)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--scaling_factor", type=float, default=2.0)

parser.add_argument("--loss_type", type=str, default="sw", choices=["sw", "gram"])
parser.add_argument("--n_iter", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1.0)

args = parser.parse_args()


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    # load exemplar
    exemplar = Image.open(args.exemplar_path)
    exemplar = preprocess_exemplar(exemplar, (args.size, args.size))
    exemplar_np = np.array(exemplar, dtype=np.float32).transpose(2, 0, 1) / 255.0

    # load VGG19 and loss function
    vgg19 = metrics.load_pretrained_VGG19_from_pth("vgg19.npy")
    if args.loss_type == "sw":
        _lossfn = metrics.create_slice_loss(vgg19, exemplar_np)
    elif args.loss_type == "gram":
        _lossfn = metrics.create_gram_loss(vgg19, exemplar_np)

    # initialize pixels
    key, subkey = jax.random.split(key)
    mean = exemplar_np.mean(axis=(1, 2), keepdims=True)
    new_size = int(args.size * args.scaling_factor)
    pixels = mean + 1e-2 * jax.random.normal(subkey, (3, new_size, new_size))

    # initialize optimizer
    optimizer = optax.lbfgs(args.lr)
    opt_state = optimizer.init(pixels)

    # define update func for each iteration
    @jax.jit
    def update(pixels, opt_state, key):
        lossfn = lambda pixels: _lossfn(pixels, key)
        loss, grads = jax.value_and_grad(lossfn)(pixels)
        updates, opt_state = optimizer.update(grads, opt_state, pixels, value=loss, grad=grads, value_fn=lossfn)
        pixels = optax.apply_updates(pixels, updates)
        return loss, pixels, opt_state

    # training loop
    for it in tqdm(range(args.n_iter), desc="iter"):
        key, subkey = jax.random.split(key)
        loss, pixels, opt_state = update(pixels, opt_state, subkey)

    # save the result using PIL
    image = Image.fromarray((np.array(pixels.clip(0.0, 1.0)).transpose(1, 2, 0) * 255).astype(np.uint8))
    image.save(f"data/result_{args.loss_type}.png")