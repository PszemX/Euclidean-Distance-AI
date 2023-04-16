import random
import numpy as np

def split_list(list_a):
    half = len(list_a) // 2
    return list_a[:half], list_a[half:]

class GeneratorI:
    def __init__(self, train_percent):
        self.train_percent = train_percent/100
        self.test_percent = (100 - train_percent)/100

    def Generator(self, size, min_range, max_range, type):
        xy = np.random.randint(min_range, max_range, size=(size, 2))
        x_train = xy[:(int)(self.train_percent*size)]
        x_test = xy[(int)(self.test_percent*size):]

        if type == "-":
            return self.GenSub(x_train, x_test)

    def GenSub(self, x_train, x_test):
        x_train_tmp = x_train[:, ::-1] # trzeba obrocic, bo inaczej dla [x, y] robi y-x zamiast x-y
        y_train = np.diff(x_train_tmp, axis=1)
        x_test_tmp = x_test[:, ::-1]
        y_test = np.diff(x_test_tmp, axis=1)
 
        return x_train, y_train, x_test, y_test

    def GenMultiply(self, x_train, x_test):
        y_train = np.prod(x_train, axis=1).reshape(-1,1)
        y_test = np.prod(x_test, axis=1).reshape(-1,1)
        return x_train, y_train, x_test, y_test
    
    def GenAdd(self, x_train, x_test):
        y_train = np.sum(x_train, axis=1, keepdims=True)
        y_test = np.sum(x_test, axis=1, keepdims=True)
 
        return x_train, y_train, x_test, y_test
    
    def GenDiverse(self, x_train, x_test):
        return 1

    def GenPower(self, x_train, x_test):
        return 1

    def GenSquare(self, x_train, x_test):
        return 1