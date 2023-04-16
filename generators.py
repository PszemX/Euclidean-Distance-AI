import random
import numpy as np

def split_list(list, percentage: float = 0.5):
    """
    Funkcja dzieli listę na dwie części na podstawie podanego procentowego podziału i zwraca dwie nowe listy.
    :param list: Lista do podziału
    :param percentage: Procentowy podział, domyślnie 0.5
    :return: Tuple dwóch list: pierwsza zawiera pierwszą część oryginalnej listy, druga - drugą część.
    """
    split_point = (len(list) * percentage).__floor__();
    return list[:split_point], list[split_point:]

class Generator:
    def __init__(self, train_percentage: str, dimensions: int = 2):
        """
        Konstruktor klasy Generator, który inicjuje wartości procentowe dla treningu i testu oraz wymiary generowanych punktów.
        :param train_percentage: Procentowy podział dla treningu jako string, np. "80%"
        :param dimensions: Liczba wymiarów dla wygenerowanych punktów.
        """
        self.train_percentage: float = float(train_percentage.replace('%', 'e-2'))
        self.test_percentage: float = float(1 - self.train_percentage)
        self.dimensions = dimensions

    def generate(self, size: int, min_range: int, max_range: int, type: str):
        """
        Funkcja generuje punkty na podstawie podanych parametrów i zwraca dwie pary list: x_train, y_train i x_test, y_test.
        :param size: Liczba punktów do wygenerowania.
        :param min_range: Minimalna wartość zakresu dla każdego wymiaru punktów.
        :param max_range: Maksymalna wartość zakresu dla każdego wymiaru punktów.
        :param type: Rodzaj operacji do wykonania na punktach. Dozwolone wartości: "substractionAB" lub "substractionBA".
        :return: Tuple dwóch par list: pierwsza zawiera punkty treningowe x_train i odpowiadające im wyniki y_train, a druga - testowe x_test i y_test.
        """
        # Generowanie losowych punktów.
        points = np.random.randint(min_range, max_range, size=(size, self.dimensions))
        results = []
        
        # Wybór rodzaju operacji i wykonanie jej.
        if type == "substractionAB":
            results = self.substract(points=[np.flip(xy) for xy in points])
        if type == "substractionBA":
            results = self.substract(points=points)
        if type == "multiplication":
            results = self.multiplication(points=points)
        if type == "addition":
            results = self.addition(points=points)
        
        # Podział listy punktów i wyników na część treningową i testową.
        x_train, x_test = split_list(list=points, percentage=self.train_percentage)
        y_train, y_test = split_list(list=results, percentage=self.train_percentage)

        return x_train, y_train, x_test, y_test
        

    def substract(self, points):
        """
        Funkcja wykonuje operację odejmowania dla podanej listy punktów.
        :param points: Lista punktów do przetworzenia.
        :return: Lista różnic między punktami, obliczonymi dla każdej pary.
        """
        return np.diff(points)

    def multiplication(self, points):
        """
        Funkcja wykonuje operację mnożenia dla podanej listy punktów.
        :param points: Lista punktów do przetworzenia.
        :return: Wynik mnożenia wszystkich punktów.
        """
        return np.multiply(points)

    def addition(self, points):
        """
        Funkcja wykonuje operację dodawania dla podanej listy punktów.
        :param points: Lista punktów do przetworzenia.
        :return: Suma wszystkich punktów.
        """
        return np.sum(points)

    def GenDiverse(self, points):
        """
        Funkcja generuje różne punkty dla każdej klasy w zbiorze treningowym i testowym.
        :param points: Lista punktów do przetworzenia.
        :return: ?.
        """
        pass

    def GenPower(self, points):
        """
        Funkcja generuje punkty o większej sile dla każdej klasy w zbiorze treningowym i testowym.
        :param points: Lista punktów do przetworzenia.
        :return: ?.
        """
        pass

    def GenSquare(self, points):
        """
        Funkcja generuje punkty o większym kwadracie dla każdej klasy w zbiorze treningowym i testowym.
        :param points: Lista punktów do przetworzenia.
        :return: ?.
        """
        pass
