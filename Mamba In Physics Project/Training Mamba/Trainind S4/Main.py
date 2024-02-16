# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:01:26 2024

@author: nicol
"""

from functools import partial

import jax
import jax.numpy as np

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve

from Functions import *

if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)
