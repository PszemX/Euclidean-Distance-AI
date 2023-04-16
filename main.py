import tensorflow as tf
import numpy as np
from generators import Generator


generator = Generator(train_percentage = "80%")
a, b, c, d = generator.generate(size = 100, min_range=0, max_range=100, type="substractionAB")
print(a)
print(b)