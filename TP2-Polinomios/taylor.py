from polis import poly as polis
from sympy import symbols, Poly

x = symbols('x')
f =3*x**3 + 2*x**2 - 5*x + 7


class taylor(polis):
    def __init__(self, f, n, x0, h=0.01, prtTaylor = True, digits = 0):
        
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
            expression += f'{self(self.x0)} '
            
            derivada = self.derivada()

            for n, coefs in enumerate(self.get_parms()):

                expression += f'+ {derivada(self.x0)} * (x - {self.x0})^{n+1} / {n+1}! '
                if len(derivada.coefs)<=1: break
                derivada = derivada.derivada()

            return expression

        return polis(self.n, self.get_parms()).get_expression() 

    def get_parms(self): return Poly(self.ft, x).all_coeffs()[::-1]
    
    def derivada_n(self,n):
        if n == 0:
            # Si n es 0, devolvemos la función original
            return self
        else:
            # Calcular el valor de h para la diferencia finita centrada

            # Definir la variable simbólica x
            x = symbols('x')

            # Calcular la derivada numérica centrada de orden n
            derivada = (self(x + n * self.h) - self(x - n * self.h)) / (2 * n * self.h)

            return derivada


#a = taylor(f,3,1)
b = polis(1,[-124,14])
print(b.get_expression())


