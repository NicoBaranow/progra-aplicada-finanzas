from polis import poly 
from sympy import symbols
import math

x = symbols('x')
f =3*x**3 + 2*x**2 - 5*x + 7

class taylor(poly):
    def __init__(self, function, n, x0, h=0.01, prtTaylor = True, digits = 0):
        
        '''
        function es una funcion. n es el grado de esa funcion. x0 es el punto a evaluar la funcion en taylor. h es el incremento de la derivada. prtTaylor dice como se imprime el polinomio.  
        '''
        self.ft = function
        self.n = n
        self.x0 = x0
        self.h = h
        self.prtTaylor = prtTaylor
        self.digits = digits
        self.feval = [self.ft(self.x0 + (n - 2*i)*self.h) for i in range(n+1)]
        self.fprime = [self.derivada_n(n) for n in range(n+1)]

        super().__init__(self.n, self.get_parms())

    def __str__(self): 
        if self.prtTaylor: 

            terms = []
            for i, coef in enumerate(self.fprime):
                terms.append(f"{round(coef, self.digits)}(x - {self.x0})^{i}/{i}!")
            return 'p(x)= ' + " + ".join(terms)

        return super().__str__()

    def derivada_n(self,n):
        '''
        Toma como valor de la derivada numérica centrada de orden n en x0
        '''
        sum = 0
        for i in range(n+1):
            sum += (-1)**i * self.combinatorial(n, i) * self.feval[i]
        return sum / (2*self.h)**n
    
    def combinatorial(self, n, k):
        '''
        Funcion auxiliar. Calcula el numero combinatorio entre n y k 
        '''
        if k == 0 or k == n: return 1
        return self.combinatorial(n-1, k-1) + self.combinatorial(n-1, k)
    
    def get_parms(self):
        aux_coeffs = [self.fprime[i] / self._factorial(i) for i in range(self.n + 1)]
        monomio = poly(1, [-self.x0, 1])
        
        for i, a_coef in enumerate(aux_coeffs):
            if i == 0:
                taylor_poly = a_coef
            else:
                taylor_poly += a_coef * monomio**i
            
        return taylor_poly.coefs     
    
    def _factorial(self, number):
        if number <= 1: return 1
        return number * self._factorial(number-1)




