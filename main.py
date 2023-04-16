import tensorflow as tf
import numpy as np
import generators


gen = generators.GeneratorI(80)
a, b, c, d = gen.Generator(100, 0, 100, "-")
print(a)
print(b)