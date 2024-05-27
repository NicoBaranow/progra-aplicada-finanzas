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
   
    def __str__(self, function = "p(x) = ", variable = 'x'):
        expression =  function

        for n, coef in enumerate(self.coefs):
            
            if coef == 0: continue
            if coef > 0 and n != 0: expression += f'+ {coef}{variable}^{n} '
            elif coef < 0: expression += f'- {coef*-1}{variable}^{n} '
            else: expression += f'{coef}{variable}^{n} '

        if expression == "p(x) = ": expression += '0'
        return expression

    def __call__(self, x):
        y = 0
        for n, coef in enumerate(self.coefs):
            y += coef * x**n
        return y

    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_poly = poly(0, [other])
            return self + new_poly
        
        elif isinstance(other, poly):
            max_degree = max(self.n, other.n)
            self_coefs = self.coefs + [0] * (max_degree - self.n)
            other_coefs = other.coefs + [0] * (max_degree - other.n)
            sum_coefs = [a + b for a, b in zip(self_coefs, other_coefs)]
            return poly(max_degree, sum_coefs)
        
        else:
            raise TypeError("Unsupported operand type for +: {}".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            new_poly = poly(0, [other])
            return self - new_poly
        
        elif isinstance(other, poly):
            max_degree = max(self.n, other.n)
            self_coefs = self.coefs + [0] * (max_degree - self.n)
            other_coefs = other.coefs + [0] * (max_degree - other.n)
            sub_coefs = [a - b for a, b in zip(self_coefs, other_coefs)]
            return poly(max_degree, sub_coefs)
        
        else:
            raise TypeError("Unsupported operand type for -: {}".format(type(other)))

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            scalar_poly = poly(0, [other])
            return self * scalar_poly
        
        elif isinstance(other, poly):
            result_coefs = [0] * (self.n + other.n + 1)
            for i in range(self.n + 1):
                for j in range(other.n + 1):
                    result_coefs[i + j] += self.coefs[i] * other.coefs[j]
            return poly(self.n + other.n, result_coefs)
        
        else:
            raise TypeError("Unsupported operand type for *: {}".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def _normalize(self):
        while self.coefs and self.coefs[-1] == 0:
            self.coefs.pop()
        self.n = len(self.coefs) - 1

    def __truediv__(self, other):
        return self._divide(other)

    def _divide(self, divisor):
        if isinstance(divisor, (int, float)):
            return self * (1/divisor), 0
        
        self._normalize()
        divisor._normalize()

        if divisor.n == 0 and divisor.coefs[-1] == 0:
            raise ZeroDivisionError("Cannot divide by a zero polynomial")

        dividend_reversed = self.coefs[::-1]
        divisor_reversed = divisor.coefs[::-1]

        quotient_coeffs_reversed = []
        remainder_reversed = dividend_reversed.copy()

        while len(remainder_reversed) >= len(divisor_reversed):
            lead_coeff = remainder_reversed[0] / divisor_reversed[0]
            quotient_coeffs_reversed.append(lead_coeff)

            product = [lead_coeff * c for c in divisor_reversed] + [0] * (len(remainder_reversed) - len(divisor_reversed))
            remainder_reversed = [rc - pc for rc, pc in zip(remainder_reversed, product)]
            remainder_reversed = remainder_reversed[1:]

        quotient_coeffs = quotient_coeffs_reversed[::-1]
        remainder_coeffs = remainder_reversed[::-1]

        while (len(quotient_coeffs) > 0) and quotient_coeffs[-1] == 0:
            quotient_coeffs.pop()
        while (len(remainder_coeffs) > 0) and remainder_coeffs[-1] == 0:
            remainder_coeffs.pop()
            
        if not quotient_coeffs:
            quotient_coeffs = [0]
        if not remainder_coeffs:
            remainder_coeffs = [0]
        
        quotient = poly(len(quotient_coeffs) - 1, quotient_coeffs)
        remainder = poly(len(remainder_coeffs) - 1, remainder_coeffs)

        return quotient, remainder

    def __floordiv__(self, other):
        quotient, _ = self._divide(other)
        return quotient

    def __rfloordiv__(self, other):
        if isinstance(other, poly):
            return other._divide(self)[0]
        else:
            return poly(n=0, coefs=[other])._divide(self)[0]

    def __mod__(self, other):
        _, remainder = self._divide(other)
        return remainder
    
    def __rmod__(self, other):
        if isinstance(other, poly):
            return other._divide(self)[1]
        else:
            return poly(n=0, coefs=[other])._divide(self)[1]

    def get_expression(self):
        return self.__str__()
    
    def poly_plot(self, a, b, **kwargs):
        x_values = np.linspace(a, b, 200)
        y_values = [self(x) for x in x_values]
        plt.plot(x_values, y_values, **kwargs)
        plt.title(f"Plot of the polynomial: {self.get_expression()}")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.show()
    
    def fprime(self, k, x0=None):
        
        if k > self.n+1: raise ValueError("El grado del polinomio a derivar +1 debe ser menor a k")
        
        fprima = self
        for i in range(k): fprima = fprima.derivada()
            
        if x0 is None: return fprima
        
        return fprima(x0)
        
    def derivada(self):
        derivative_coefs = [self.coefs[i] * i for i in range(1, self.n + 1)]
        return poly(self.n - 1, derivative_coefs)

    def root_find_bisection(self, a=-200, b=100, tolerance=1e-5, iter=100000):
        iter_count = 0
        if self(a) * self(b) > 0:
            raise ValueError("No se puede garantizar la existencia de una raíz en el intervalo dado.")

        while iter_count < iter:
            root = (a+b) / 2
            if abs(self(root)) <= tolerance: return round(root, 3)
            if self(a) * self(root) < 0: b = root
            else: a = root
            iter_count += 1
        
        print(f"El método de bisección no convergió después de {iter} iteraciones.")
        return None

    def root_find_newton(self, x0=-100, tolerance=1e-5, iter=100000):
        iter_count = 0
        root = x0
        while iter_count < iter:
            f_x = self(root)

            if abs(f_x) < tolerance: return round(root, 3)
            
            f_prime_x = self.derivada()(root)
            if f_prime_x == 0:
                raise ValueError(f"Derivada en el punto {root} es cero.")
            root = root - f_x / f_prime_x
            iter_count += 1

        print(f"El método de Newton-Raphson no convergió después de {iter} iteraciones.")
        return None

    def root_find_secante(self, x0=-1, x1=2, tolerance=1e-5, max_iter=1000):
        iterations = 0

        while iterations < max_iter:
            if abs(self(x1) - self(x0)) < 1e-10:
                raise ValueError("La diferencia entre los valores de la función en x1 y x0 es demasiado pequeña.")

            root = x0 - self(x0) * (x1 - x0) / (self(x1) - self(x0))

            if abs(self(root)) < tolerance: return round(root, 3)
            x0 = x1
            x1 = root

            iterations += 1

        print(f"No se encontró raíz entre {x0} y {x1} después de {max_iter} iteraciones.")
        return None

    def find_roots(self, tolerance=1e-5):
        roots = []
        residual_poly = self
    
        while residual_poly.n > 0:
            root = residual_poly.root_find_secante()
            if not root: break
            
            multiplicity = 0

            while True:
                cociente = residual_poly // poly(1, [-root, 1])
                resto = residual_poly % poly(1, [-root, 1])

                if resto.n == 0 and abs(resto.coefs[0]) < tolerance:
                    multiplicity += 1
                    residual_poly = cociente
                else: break

            if multiplicity > 0: roots.append((round(root, 3), multiplicity))
        
        return roots

    def factorize(self):
        roots = self.find_roots()
        expression = ''
        for root, multiplicity in roots: expression += f"(x - {root})^{multiplicity} "
        return expression