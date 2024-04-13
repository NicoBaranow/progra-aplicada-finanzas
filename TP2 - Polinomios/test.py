
import matplotlib.pyplot as plt
import numpy as np

n = 3 #grado polinomio
coefs = [2,3,0,-5] #2 -2x^1 

class poly:
    def __init__(self, n=0, coefs = [0]):
        '''
        n: int igual al grado del polinomio
        coefs: list con los coeficientes del polinomio. Coefs[0] corresponde a la potencia 0, coefs[1] a la potencia 1 y asÌ sucesivament
        '''
        if n != len(coefs)-1:
            raise ValueError("El grado del polinomio debe ser mayor o igual al número de coeficientes proporcionados.")
        self.n = n
        self.coefs = coefs

    def __call__(self, x):
        y = 0
        for n, coef in enumerate(self.coefs):
            y += coef * x**n
        return y
    
    def __add__(self, other):
        if isinstance(other, (int, float)): #si other es int o floar, devuelve True
            # Crear un polinomio de grado 0 con el valor del escalar
            new_poly = poly(0, [other])
            return self + new_poly
        
        elif isinstance(other, poly):
            # Extender el polinomio de menor grado al de mayor grado
            max_degree = max(self.n, other.n)
            self_coefs = self.coefs + [0] * (max_degree - self.n)
            other_coefs = other.coefs + [0] * (max_degree - other.n)
            # Sumar los coeficientes de los polinomios extendidos
            sum_coefs = [a + b for a, b in zip(self_coefs, other_coefs)]
            
            return poly(max_degree, sum_coefs)
        
        else:
            raise TypeError("Unsupported operand type for +: {}".format(type(other)))
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)): #si other es int o floar, devuelve True
            # Crear un polinomio de grado 0 con el valor del escalar
            new_poly = poly(0, [other])
            return self - new_poly
        
        elif isinstance(other, poly):
            # Extender el polinomio de menor grado al de mayor grado
            max_degree = max(self.n, other.n)
            self_coefs = self.coefs + [0] * (max_degree - self.n)
            other_coefs = other.coefs + [0] * (max_degree - other.n)
            # Sumar los coeficientes de los polinomios extendidos
            sub_coefs = [a - b for a, b in zip(self_coefs, other_coefs)]
            
            return poly(max_degree, sub_coefs)
        
        else:
            raise TypeError("Unsupported operand type for +: {}".format(type(other)))
        
    def __rsub__(self, other):
        return self.__sub__(other)

    def get_expression(self):
        expression = "p(x) = "

        for n, coef in enumerate(self.coefs):
            
            if coef == 0: continue
            if coef > 0 and n != 0: expression += f'+ {coef}x^{n} '
            elif coef < 0: expression += f'- {coef*-1}x^{n} '
            else: expression += f'{coef}x^{n} '

        return expression
    
    def poly_plot(self, a, b, **kwargs):
        x_values = np.linspace(a, b, 100000)
        y_values = self(x_values)
        plt.plot(x_values, y_values, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Grafico del polinomio')
        plt.grid(True)
        plt.show()

a = poly(n, coefs)
b = poly(5,[2,51,51,4,6,32])

newPoly =  b-a
print(newPoly.get_expression())
