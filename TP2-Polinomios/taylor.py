from polis import poly as polis
from sympy import symbols, Poly

x = symbols('x')
f = 3*x**3 + 2*x**2 - 5*x + 7


class taylor(polis):
    def __init__(self, f, n, x0, h=0.01, prtTaylor = False, digits = 0):
        
        '''
        f es una funcion. n es el grado de esa funcion. x0 es el punto a evaluar la funcion en taylor. h es el incremento de la derivada. prtTaylor dice como se imprime el polinomio.  
        '''

        self.ft = f
        self.n = n
        self.x0 = x0
        self.h = h
        # self.feval = [self.ft(self.x0 + (self.n - 2*i)*self.h) for i in range(self.n + 1)]
        # self.fprime = [self.derivada_n(j) for j in range(self.n + 1)]
        self.prtTaylor = prtTaylor
        self.digits = digits
        super(taylor, self).__init__(self.n, self.get_parms())

    def __str__(self): 
        if self.prtTaylor: 
            expression = f'P(x,{self.x0}) = '
            expression += str(f(self.x0))
            for i in len(self.get_parms()):
                derivada = self.derivada()
                derivada

            

        return polis(self.n, self.get_parms()).get_expression() 

    def get_parms(self): return Poly(self.ft, x).all_coeffs()[::-1]
        
a = taylor(f,3,1)
print(a)


### JUAN ###
# class taylor(poly):
#     def __init__(self, ft, n, x0, feval = [], fprime = [], h=0.01, prtTaylor = False, digits = 0):
        
#         self.ft = ft
#         self.n = n
#         self.x0 = x0
#         self.h = h
#         self.prtTaylor = prtTaylor
#         self.digits = digits
#         self.feval = feval
#         self.fprime = fprime

#         for i in range(self.n+1): 
#             self.feval.append(self.ft(self.x0 + (self.n - 2*i)*self.h))
        
#         for i in range(self.n + 1):
#             self.fprime.append(self.derivada_n(i))
        
#         self.poly_taylor = poly(self.n, self.get_parms())

### FINI ###
# class taylor(poly):
#     def __init__(self, ft, n, x0, digits, h=0.01):
        
#         super(taylor, self.__init__(self.get_parms))
#         self.ft = ft
#         self.n = n
#         self.x0 = x0
#         self.h = h
#         self.feval = [self.ft(self.x0 + (self.n-i) * self.h) for i in range(self.n + 1)]
#         self.fprime = [self.derivada_n(j) for j in range(self.n + 1)]
#         self.prtTaylor = True
#         self.digits = digits

