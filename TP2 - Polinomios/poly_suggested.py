import matplotlib.pyplot as plt
import numpy as np
import random

class poly:
    def __init__(self, n=0, coefs=[0]):

        if len(coefs) != n + 1:
            raise ValueError("The number of coefficients must be equal to the degree of the polynomial + 1.")

        self.n = n
        self.coefs = coefs

    def get_expression(self):
        terms = []
        for i, coef in enumerate(self.coefs):
            if abs(coef) >= 1e-5:
                sign = "+" if coef >= 0 else "-"
                coef_r = round(coef, 3)
                abs_coef = abs(coef_r)
                term = f"{sign} {abs_coef}x^{i}" if i != 0 else str(coef_r)
                terms.append(term)
        expression = " ".join(terms)
        if expression[0] == "+":
            expression = expression[2:]
        return expression

    def __call__(self, x):
        result = 0
        for i, coef in enumerate(self.coefs):
            result += coef * (x ** i)
        return result

    def __str__(self):
        return self.get_expression()
    
    def poly_plt(self, a, b, **kwargs):
        x_values = np.linspace(a, b, 200)
        y_values = [self(x) for x in x_values]
        plt.plot(x_values, y_values, **kwargs)
        plt.title(f"Plot of the polynomial: {self.get_expression()}")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.show()
               
    def __add__(self, other):
        if isinstance(other, poly):
            max_degree = max(self.n, other.n)
            new_coefs = [0] * (max_degree + 1)
            for i in range(max_degree + 1):
                coef_self = self.coefs[i] if i <= self.n else 0
                coef_other = other.coefs[i] if i <= other.n else 0
                new_coefs[i] = coef_self + coef_other
            return poly(max_degree, new_coefs)
        elif isinstance(other, (int, float)):
            new_poly = poly(0, [other])
            return self + new_poly
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, poly):
            max_degree = max(self.n, other.n)
            new_coefs = [0] * (max_degree + 1)
            for i in range(max_degree + 1):
                coef_self = self.coefs[i] if i <= self.n else 0
                coef_other = other.coefs[i] if i <= other.n else 0
                new_coefs[i] = coef_self - coef_other
            return poly(max_degree, new_coefs)
        elif isinstance(other, (int, float)):
            new_poly = poly(0, [other])
            return self - new_poly
        else:
            return NotImplemented

    def __rsub__(self, other):
        # Handle scalar subtraction where 'other' is a scalar
        if isinstance(other, (int, float)):
            new_poly = poly(0, [other])
            return new_poly - self
        
        # Handle polynomial subtraction where 'other' is a poly instance
        elif isinstance(other, poly):
            max_degree = max(self.n, other.n)
            new_coefs = [0] * (max_degree + 1)
    
            for i in range(max_degree + 1):
                coef_self = self.coefs[i] if i <= self.n else 0
                coef_other = other.coefs[i] if i <= other.n else 0
                new_coefs[i] = coef_other - coef_self
    
            return poly(max_degree, new_coefs)
        
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, poly):
            # Polynomial multiplication
            new_degree = self.n + other.n
            new_coefs = [0] * (new_degree + 1)
            for i in range(self.n + 1):
                for j in range(other.n + 1):
                    new_coefs[i + j] += self.coefs[i] * other.coefs[j]
            return poly(new_degree, new_coefs)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            new_poly = poly(0, [other])
            return self * new_poly
        else:
            return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, potencia):
        if not isinstance(potencia, int):
            raise ValueError("The exponent must be an integer.")
        
        if potencia == 0:
            return poly(0, [1])
        
        if potencia < 0:
            raise ValueError("Negative powers are not supported for polynomials in this implementation.")
        
        salida = poly(self.n, self.coefs.copy())
        for _ in range(1, potencia):
            salida = salida * self
        return salida   
    
    def _normalize(self):
        while self.coefs and self.coefs[-1] == 0:
            self.coefs.pop()
        self.n = len(self.coefs) - 1
        
    def __truediv__(self, other):
        return self._divide(other)

    def _divide(self, divisor):
        if isinstance(divisor, (int, float)):
            return self * (1/divisor), 0
        # Normalize the polynomials to remove trailing zeros
        self._normalize()
        divisor._normalize()

        if divisor.n == 0 and divisor.coefs[-1] == 0:
            raise ZeroDivisionError("Cannot divide by a zero polynomial")

        # Reverse the coefficients for the division operation
        dividend_reversed = self.coefs[::-1]
        divisor_reversed = divisor.coefs[::-1]

        quotient_coeffs_reversed = []
        remainder_reversed = dividend_reversed.copy()

        while len(remainder_reversed) >= len(divisor_reversed):
            lead_coeff = remainder_reversed[0] / divisor_reversed[0]
            quotient_coeffs_reversed.append(lead_coeff)

            # Subtract the product of the divisor and the current quotient term from the remainder
            product = [lead_coeff * c for c in divisor_reversed] + [0] * (len(remainder_reversed) - len(divisor_reversed))
            remainder_reversed = [rc - pc for rc, pc in zip(remainder_reversed, product)]

            # Remove the used highest degree term from remainder
            remainder_reversed = remainder_reversed[1:]

        # Reverse the quotient and remainder back to original coefficient order
        quotient_coeffs = quotient_coeffs_reversed[::-1]
        remainder_coeffs = remainder_reversed[::-1]

        # Ensure that we remove any leading zeros after reversal
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
            
    def fprime(self, k, x0=None):
        if k < 0:
            raise ValueError("Derivative order must be non-negative")

        result = self
        for _ in range(k):
            derivative_coefs = [i * result.coefs[i] for i in range(1, len(result.coefs))]
            result = poly(len(derivative_coefs) - 1, derivative_coefs)

        if x0 is not None:
            return result(x0)
        else:
            return result

    def rootfind(self, method='newton', **kwargs):
        if method == 'bisection':
            root_finder = BisectionMethod()
            return root_finder.find_root(self.__call__, **kwargs)
        elif method == 'newton':
            root_finder = NewtonMethod()
            derivative_func = lambda x: self.fprime(1)(x)
            return root_finder.find_root(self.__call__, derivative_func, **kwargs)
        else:
            raise ValueError("Unsupported root finding method.")
            
    def findroots(self, method = 'newton', **kwargs):
        roots = []
        residual_poly = self
    
        while residual_poly.n > 0:
            root = residual_poly.rootfind(method, **kwargs)        
            if not root:
                break
            
            multiplicity = 0

            while True:
                quotient, remainder = residual_poly._divide(poly(1, [-root, 1]))

                if remainder.n == 0 and abs(remainder.coefs[0]) < 1e-3:
                    multiplicity += 1
                    residual_poly = quotient
                else:
                    break

            if multiplicity > 0:
                roots.append((root, multiplicity))
        
        return roots, residual_poly

    
    def factorize(self, method = 'newton', **kwargs):
        roots, residual_poly = self.findroots(method, **kwargs)
        factors = []
    
        for root, multiplicity in roots:
            factors.append(f"(x - {round(root, 3)})^{multiplicity}")
    
        if residual_poly.n > 0 or (residual_poly.n == 0 and residual_poly.coefs[0] != 0):
            factors.append(f"({str(residual_poly)})")
    
        return " * ".join(factors)



#%%      
class BisectionMethod:
    def find_root(self, func, a = -1_000_000, b = 1_000_000, x_tolerance=1e-5, y_tolerance=1e-5, max_iterations=100000):
        if func(a) * func(b) >= 0:
            raise ValueError("The function must have different signs at the endpoints a and b.")

        for _ in range(max_iterations):
            mid = (a + b) / 2.0  # Compute the midpoint in each iteration
            f_mid = func(mid)
            
            if abs(f_mid) < y_tolerance or (b - a) / 2 < x_tolerance:
                return mid
            if f_mid * func(a) < 0:
                b = mid
            else:
                a = mid

        return None


class NewtonMethod:
    def find_root(self, func, derivative, x0 = 0.01, x_tolerance=1e-5, y_tolerance = 1e-5, max_iterations=100000, x_step = 1_000):
        x = x0
        for i in range(max_iterations):
            f_x = func(x)
            df_x = derivative(x)
            
            if abs(f_x) < y_tolerance:
                return x

            if df_x == 0:
                raise ValueError(f"Derivative is zero at x = {x}, cannot continue Newton's method")
            
            x_new = x - f_x / df_x
            if abs(x_new - x) < x_tolerance:
                return self.find_root(func, derivative, x0 = random.uniform(x0-x_step, x0+x_step), 
                                      x_tolerance = x_tolerance, y_tolerance = y_tolerance, 
                                      max_iterations = max_iterations - i - 1, x_step = x_step)
            x = x_new

        return None

class linear(poly):
    def __init__(self, a, b):
        super().__init__(n=1, coefs=[b, a])

    def findroots(self, method=None, **kwargs):
        # For a linear polynomial ax + b, the root is -b/a
        if self.coefs[1] != 0:
            return [(-self.coefs[0] / self.coefs[1], 1)], poly(0, [0])
        else:
            return [], poly(0, [self.coefs[0]])

    def factorize(self, method=None, **kwargs):
        if self.coefs[1] == 0:  # a = 0, hence not a valid linear equation
            return str(self)
        root = -self.coefs[0] / self.coefs[1]
        return f"(x - {round(root, 3)})"


class quadratic(poly):
    def __init__(self, a, b, c):
        super().__init__(n=2, coefs=[c, b, a])

    def findroots(self, method=None, **kwargs):
        a, b, c = self.coefs[2], self.coefs[1], self.coefs[0]
        discriminant = b**2 - 4*a*c

        if a == 0:  
            return linear(b, c).findroots()

        if discriminant > 0:
            root1 = (-b + np.sqrt(discriminant)) / (2*a)
            root2 = (-b - np.sqrt(discriminant)) / (2*a)
            return [(root1, 1), (root2, 1)], poly(0, [0])
        elif discriminant == 0:
            root = -b / (2*a)
            return [(root, 2)], poly(0, [0])
        else:
            return [], poly(2, [c, b, a])

    def factorize(self, method=None, **kwargs):
        roots, _ = self.findroots()
        if not roots:
            return str(self)
        elif len(roots) == 1:
            root, multiplicity = roots[0]
            return f"(x - {round(root, 3)})^2"
        else:
            root1, root2 = roots[0][0], roots[1][0]
            return f"(x - {round(root1, 3)})(x - {round(root2, 3)})"
        
  
#%%
class Taylor(poly):
    def __init__(self, function, N, x0, h=0.001, prtTaylor = True):
        self.fT = function
        self.N = N
        self.x0 = x0
        self.h = h
        self.prtTaylor = prtTaylor
        self.digits = 3

        self.feval = [self.fT(self.x0 + (N - 2*i)*self.h) for i in range(N+1)]
        self.fprime = [self.derivada_n(n) for n in range(N+1)]

        super().__init__(self.N, self.get_parms())

    def __str__(self):
        if self.prtTaylor:
            terms = []
            for i, coef in enumerate(self.fprime):
                terms.append(f"{round(coef, self.digits)}(x - {self.x0})^{i}/{i}!")
            return 'p(x)= ' + " + ".join(terms)
        else:
            return super().__str__()

    def derivada_n(self, n):
        sum = 0
        for i in range(n+1):
            sum += (-1)**i * self.combinatorial(n, i) * self.feval[i]
        return sum / (2*self.h)**n

    def combinatorial(self, n, k):
        if k == 0 or k == n: return 1
        return self.combinatorial(n-1, k-1) + self.combinatorial(n-1, k)

    def get_parms(self):
        aux_coeffs = [self.fprime[i] / self._factorial(i) for i in range(self.N + 1)]
        monomio = poly(1, [-self.x0, 1])
        
        for i, a_coef in enumerate(aux_coeffs):
            if i == 0:
                taylor_poly = a_coef
            else:
                taylor_poly += a_coef * monomio**i
            
        return taylor_poly.coefs

    def _factorial(self, number):
        if number <= 1:
            return 1
        else:
            return number * self._factorial(number-1)


#%%

class Lagrange(poly):
    def __init__(self, data):
        '''
        Recibe una lista de tuplas con los puntos a interpolar.
        '''
        self.data = data
        self.x_values = [tupla[0] for tupla in self.data]
        self.y_values = [tupla[1] for tupla in self.data]
        self.lagrange_coefs = self._get_lagrange_coefs()
        self.lagrange_grade = len(self.lagrange_coefs) - 1
        super().__init__(self.lagrange_grade, self.lagrange_coefs)
    
    def _get_lagrange_coefs(self):
        
        lagrange_poly = 0
        for i, yi in enumerate(self.y_values):
            wi = 1 
            xi = self.x_values[i]
            
            for j, xj in enumerate(self.x_values):
                if xi == xj:
                    continue
                numerator = poly(n=1, coefs=[-xj, 1]) # x - xj
                denominator = xi - xj
                monomial = numerator // denominator
                wi *= monomial
                
            lagrange_poly += yi * wi
        
        return lagrange_poly.coefs
    
    
    def poly_plt(self, a, b, **kwargs):
        new_x_values = np.linspace(a, b, 100)
        new_y_values = [self(x) for x in new_x_values]
        plt.plot(new_x_values, new_y_values, **kwargs)
        plt.scatter(self.x_values, self.y_values)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.show()
        
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
        plt.plot(self.x_values, self.y_values_interpolated, color = 'red', label = str(self))
        plt.legend()
        
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
        root_finder = NewtonMethod()
        numeric_beta = root_finder.find_root(self.derivate1_square_sum_function, self.derivate2_square_sum_function)
        return numeric_beta




