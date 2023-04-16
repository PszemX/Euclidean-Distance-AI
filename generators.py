import random
import numpy as np

def split_list(list, percentage: float = 0.5):
    split_point = (len(list) * percentage).__floor__();
    return list[:split_point], list[split_point:]

class Generator:
    def __init__(self, train_percentage: str, dimensions: int = 2):
        self.train_percentage: float = float(train_percentage.replace('%', 'e-2'))
        self.test_percentage: float = float(1 - self.train_percentage)
        self.dimensions = dimensions

    def generate(self, size: int, min_range: int, max_range: int, type: str):
        points = np.random.randint(min_range, max_range, size=(size, self.dimensions))
        results = []
        if type == "substractionAB":
            results = self.substract([np.flip(xy) for xy in points])
        if type == "substractionBA":
            results = self.substract(points)
        
        x_train, x_test = split_list(list=points, percentage=self.train_percentage)
        y_train, y_test = split_list(list=results, percentage=self.train_percentage)

        return x_train, y_train, x_test, y_test
        

    def substract(self, points):
        return np.diff(points)

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