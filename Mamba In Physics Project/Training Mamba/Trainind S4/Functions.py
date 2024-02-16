# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:23:16 2024

@author: nicol
"""

##A place to define functions that will be called In the main.

from functools import partial

import jax
import jax.numpy as np

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve

def random_SSM(rng,N):
    a_r, b_r, c_r=jax.random.split(rng,3)
    A=jax.random.uniform(a_r,(N,N))
    B=jax.random.uniform(b_r,(N,1))
    C=jax.random.uniform(c_r,(1,N))
    
    return A, B, C