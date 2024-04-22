import matplotlib.pyplot as plt
import numpy as np

# b = poly(2, [-2,0,3])
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

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Crear un polinomio de grado 0 con el valor del escalar
            scalar_poly = poly(0, [other])
            # Multiplicar el polinomio de grado 0 por el polinomio actual
            return self * scalar_poly
        
        elif isinstance(other, poly):
            # Multiplicación de polinomio por otro polinomio
            result_coefs = [0] * (self.n + other.n + 1)
            for i in range(self.n + 1):
                for j in range(other.n + 1):
                    result_coefs[i + j] += self.coefs[i] * other.coefs[j]
            return poly(self.n + other.n, result_coefs)
        
        else:
            raise TypeError("Unsupported operand type for *: {}".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def get_expression(self):
        expression = "p(x) = "

        for n, coef in enumerate(self.coefs):
            
            if coef == 0: continue
            if coef > 0 and n != 0: expression += f'+ {coef}x^{n} '
            elif coef < 0: expression += f'- {coef*-1}x^{n} '
            else: expression += f'{coef}x^{n} '

        if expression == "p(x) = ": expression += '0'
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
    
    def derivada (self):
        derivative_coefs = [self.coefs[i] * i for i in range(1, self.n + 1)]
        return poly(self.n - 1, derivative_coefs)

    def root_find_bisection(self,a,b, tolerance = 1e-5, iter = 1000):
   
        '''
        Toma como parametro dos valores entre los cuales buscar la raiz real del polinomio; tolerancia del 0; iteraciones.
        a*b <= 0. 
        Devuelve una raiz real del polinomio. Caso contrario, da error
        '''
        iter_count = 0
        if self(a) * self(b) > 0: raise ValueError("No se puede garantizar la existencia de una raíz en el intervalo dado.")

        while iter_count < iter:
            c = (a+b) / 2
            if abs(c) < tolerance: return c
            elif self(a) * self(c) < 0: b = c
            else: a = c
            iter_count += 1
        
        raise ValueError(f"El método de bisección no convergió después de {iter} iteraciones.")
    
    def root_find_newton(self, x0, tolerance = 1e-5, iter = 1000):
        iter_count = 0
        x = x0
        while iter_count < iter:
            f_x = self(x)

            if abs(f_x) < tolerance:
                return x
            
            f_prime_x = self.derivada()(x)
            if f_prime_x == 0:
                raise ValueError(f"Derivada en el punto {x} es cero.")
            x = x - f_x / f_prime_x
            iter_count += 1

        raise ValueError(f"El método de Newton-Raphson no convergió después de {iter} iteraciones.")

    def findroots(self):
        roots = []
        residual_poly = self
        while True:
            try:
                # Buscar una raíz utilizando el método de Newton-Raphson
                root = residual_poly.root_find_newton(0)
                # Determinar la multiplicidad de la raíz
                multiplicity = 1
                while True:
                    # Dividir el polinomio residual por el monomio (x - root)
                    divisor = poly(1, [-root, 1])
                    residual_poly = residual_poly // divisor
                    # Si el residuo no es cero, la raíz tiene una multiplicidad mayor
                    if residual_poly != poly(0):
                        multiplicity += 1
                    else:
                        break
                # Agregar la raíz y su multiplicidad a la lista de raíces
                roots.append((root, multiplicity))
            except ValueError:
                # Si no se puede encontrar más raíces, terminar el bucle
                break
        return roots, residual_poly
    
    def fprime(self, k, x0 = None):
        
        if k > self.n+1: raise ValueError("El grado del polinomio a derivar +1 debe ser menor a k")
        
        fprima = self
        for i in range(k):
            fprima = fprima.derivada()
            
        if x0 == None: return fprima
        
        return fprima(x0)
        
 #Missing __floordiv__; __rfloordiv__; __mod__; __rmod__; find_roots.


class linreg(poly):

    def __init__(self, datos = []):
        '''
        
        '''
        self.datos = datos
        self.beta = np.sum((self.X - np.mean(self.X)) * (self.Y - np.mean(self.Y))) / np.sum((self.X - np.mean(self.X))**2)
        self.alpha = alpha
        super(linreg, self).__init__()
