import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('TP2-Polinomios')

from polis import poly

class linreg(poly):
    
    def __init__(self, data):
        self.data = data
        self.x_values = [tupla[0] for tupla in self.data]
        self.y_values = [tupla[1] for tupla in self.data]
        
        self.x_mean = sum(self.x_values) / len(self.x_values)
        self.y_mean = sum(self.y_values) / len(self.y_values)

        self.beta = self._calculate_beta()
        self.alpha = self._calculate_alpha()
        
        super().__init__(n=1, coefs=[self.alpha, self.beta])
        
    def _calculate_beta(self):

        numerator = 0
        denominator = 0
        for x, y in self.data:
            numerator += ((x - self.x_mean) * (y - self.y_mean))
            denominator += ((x - self.x_mean)**2)
            
        beta = numerator / denominator
 
        return beta
                    
    def _calculate_alpha(self):
        return self.y_mean - self.beta * self.x_mean

    def __str__(self):
        return f"{round(self.alpha, 4)} + {round(self.beta, 4)} * x"
    
    def regplot(self):
        self.y_values_interpolated = self._interpolate(self.x_values)

        plt.scatter(self.x_values, self.y_values)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.plot(self.x_values, self.y_values_interpolated, color = 'red', label = str(self))
        plt.legend()
        plt.show()
        
    def _interpolate(self, new_x_values):
        new_y_values = [self(x) for x in new_x_values]
        return new_y_values
    
    def square_sum_function(self, beta):
        return sum([(y - (self.alpha + beta * x))**2 for x, y in self.data])
        
    def derivate1_square_sum_function(self, beta, h = 0.0001):
        forward_step = self.square_sum_function(beta + h)
        backward_step = self.square_sum_function(beta - h)
        return (forward_step - backward_step) / (2*h)
    
    def derivate2_square_sum_function(self, beta, h = 0.0001):
        forward_step = self.derivate1_square_sum_function(beta + h)
        backward_step = self.derivate1_square_sum_function(beta - h)
        return (forward_step - backward_step) / (2*h)
            
    def NR_reg(self):
        numeric_beta = self.root_find_newton()
        # root_finder = NewtonMethod()
        # numeric_beta = root_finder.find_root(self.derivate1_square_sum_function, self.derivate2_square_sum_function)
        return numeric_beta

a = linreg([(0, 2), (2, 3), (4, 8), (3, 4),(1, 1)])
