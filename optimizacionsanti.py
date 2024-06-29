
import matplotlib.pyplot as plt
import random
import sys
import matplotlib.pyplot as plt
sys.path.append('TP4 - matrices')
sys.path.append('TP2 - Polinomios')
sys.path.append('TP6 - Vectores')
import poly_suggested 
from matrices import matrix
from vectores import vector
from polis import poly
import numpy as np


#%%
class vector(matrix):
    def __init__(self, x):
        self.eps = 0.00001
        self.x = x
        super().__init__(x, len(x), 1)

    def __sub__(self, other):
        return vector([a - b for a, b in zip(self.x, other.x)])

    def __add__(self, other):
        return vector([a + b for a, b in zip(self.x, other.x)])

    def __mul__(self, scalar):
        return vector([a * scalar for a in self.x])

    def norm(self):
        return np.linalg.norm(self.x)

    def __getitem__(self, index):
        return self.x[index]

    def __setitem__(self, index, value):
        self.x[index] = value
        
    def inner(self, other):
        # Calcula el producto interno entre dos vectores
        if not isinstance(other, vector):
            raise TypeError('Se espera un vector')
        if other.r != self.r or other.c != 1:
            raise TypeError('Los vectores deben tener la misma cantidad de elementos')
        resultado = self.transpose() * other
        return resultado.elems[0]
    
    def outer(self, other):
        # Calcula el producto externo entre dos vectores
        resultado = self * other.transpose()
        return resultado
    
    def norm2(self):
        # Calcula la norma al cuadrado del vector
        resultado = self.inner(self)
        return resultado
    
    def norm(self):
        # Calcula la norma del vector
        resultado = self.norm2()**0.5
        return resultado

    def versor(self):
        # Calcula el versor (vector unitario) del vector
        norma = self.norm()
        if norma == 0:
            raise ValueError('La norma es 0')
        else:
            resultado = self * (1 / norma)
        return resultado
    
    def proyect(self, b):
        # Proyecta el vector sobre otro vector b
        if not isinstance(b, vector):
            raise TypeError('b debe ser un vector')
        b_versor = b.versor()
        producto_interno = self.inner(b_versor)
        resultado = producto_interno * b_versor
        return producto_interno, b_versor, resultado
    
    def orth(self, b):
        # Calcula la componente ortogonal del vector respecto a b
        proyecciones = self.proyect(b)
        proyeccion = proyecciones[2]
        w = self - proyeccion
        return w
        
    def __mul__(self, other):
        # Multiplica el vector por un escalar o una matriz
        if isinstance(other, (int, float)):
            resultado = super().__mul__(other)
            return vector(resultado.elems)
        else:
            return super().__mul__(other)
        
    def __add__(self, other):
        # Suma el vector con otro vector o matriz
        resultado = super().__add__(other)
        return vector(resultado.elems)
        
    def __radd__(self, other):
        # Suma el vector con otro vector o matriz (para soportar la suma con escalar)
        resultado = super().__radd__(other)
        return vector(resultado.elems)
        
    def __sub__(self, other):
        # Resta el vector con otro vector o matriz
        resultado = super().__sub__(other)
        return vector(resultado.elems)
    
    def __rsub__(self, other):
        # Resta el vector con otro vector o matriz (para soportar la resta con escalar)
        resultado = super().__rsub__(other)
        return vector(resultado.elems)

    def orth_proj(self, lista):
        # Computa la proyección ortogonal a los vectores de la lista usando el proceso de Gram-Schmidt
        import sys        
        N = len(lista)
        lista_norm = [lista[0].versor()]
        if N > 1:
            for i in range(1, len(lista)):
                v = lista[i]
                for j in lista_norm:                    
                    v = v - j.inner(v) * j
                if v.norm() < self.eps:
                    sys.exit('En la ortogonalizacion, uno de los vectores no es independiente')
                else:
                    v = v.versor()
                lista_norm.append(v)
        v = self
        for j in lista_norm:
            v = v - v.inner(j) * j
        return v, lista_norm
#%%           
class funcion(object):
    def __init__(self, f, nd, name="None"):   
        # Inicializa la función con su definición, número de variables y nombre
        self.f = f
        self.nd = nd
        self.name = name
    
    def __call__(self, x): 
        # Evalúa la función con un vector de entrada x
        if len(x.x) == self.nd:
            salida = self.f(x.x)
        else:
            print("La función " + self.name + " requiere inputs de dimensión " + str(self.nd))
            salida = None
        return salida
    
    def grad(self, x, h=0.001): 
        # Calcula el gradiente de la función en el punto x
        salida = [0] * self.nd
        for i in range(self.nd):
            aux = [0] * self.nd 
            aux[i] = h
            salida[i] = (self.f((x + vector(aux)).x) - self.f((x - vector(aux)).x)) / (2 * h)
        return vector(salida)

class optnd(object):
    def __init__(self, f, h=0.001):
        # Inicializa la clase de optimización en n dimensiones con la función f y el paso h
        self.f = f
        self.h = h
        
    def derivada_parcial(self, x, i):
        # Calcula la derivada parcial de la función respecto a la i-ésima variable
        aux = [0] * len(x.x)
        aux[i] = self.h
        return (self.f(vector([xj + auxj for xj, auxj in zip(x.x, aux)])) - self.f(vector([xj - auxj for xj, auxj in zip(x.x, aux)]))) / (2 * self.h)
    
    def gradf(self, x):
        # Calcula el gradiente de la función en el punto x
        return vector([self.derivada_parcial(x, i) for i in range(len(x.x))])
 
    def fv(self, x, v):
        # Calcula la derivada direccional de la función en la dirección del vector v
        gx = self.gradf(x)
        return gx.inner(v)
        
    def campo_gradiente(self, xlim, ylim, nx, ny, arrow_length=0.04, arrow_width=0.01, head_width=0.1, color='black', x=False):
        # Dibuja el campo de gradiente de la función en el rango dado
        hx = (xlim[1] - xlim[0]) / nx
        hy = (ylim[1] - ylim[0]) / ny
        
        plt.figure(figsize=(10, 6))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Campo de gradiente')
        plt.grid(True)
        
        for i in range(nx):
            x = xlim[0] + i * hx
            for j in range(ny):
                y = ylim[0] + j * hy
                dx, dy = self.gradf(vector([x, y])).x
                plt.arrow(x, y, dx * arrow_length, dy * arrow_length, width=arrow_width, head_width=head_width, head_length=0.1, color=color)
        if not x:        
            plt.show()
        
    def hessian(self, x):
        # Calcula la matriz Hessiana de la función en el punto x
        n = len(x.x)
        hess = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    hess[i][j] = (self.f(vector([xk + (self.h if k == i else 0) for k, xk in enumerate(x.x)])) - 2 * self.f(x) + self.f(vector([xk - (self.h if k == i else 0) for k, xk in enumerate(x.x)]))) / (self.h ** 2)
                else:
                    hess[i][j] = (self.f(vector([xk + (self.h if k == i else 0) + (self.h if k == j else 0) for k, xk in enumerate(x.x)])) - self.f(vector([xk + (self.h if k == i else 0) for k, xk in enumerate(x.x)])) - self.f(vector([xk + (self.h if k == j else 0) for k, xk in enumerate(x.x)])) + self.f(x)) / (4 * self.h ** 2)
        return matrix([item for sublist in hess for item in sublist], n, n)
    
    def gdescent(self, x0, delta=0.01, tol=0.001, Nmax=10000, graficar=False): 
        # Realiza el descenso por gradiente para minimizar la función
        x = x0
        history = []
        for n in range(Nmax):
            x_original = x
            fx = self.gradf(x)
            x = x - delta * fx
            history.append(x.x)
            distancia = ((x - x_original).norm2()) ** 0.5
            if distancia < tol:
                break
        if graficar:
            return history, (x.x, self.f(x))
        else:
            return x, self.f(x)

    def construir_hessiana(self, x):
        # Construye la matriz Hessiana de la función en el punto x
        return self.hessian(x)

class optim():
    def __init__(self, f, h=[], g=[]): 
        # Inicializa la clase de optimización con la función objetivo y las restricciones
        self.f = f
        self.h = h
        self.g = g
        self.minimize = True
        self.coef = -1.0
        self.learn = 0.1
        self.x = None
        
    def solve(self, x0, minimize=True, learning_rate=0.1, tol=0.001, Nmax=10000): 
        # Resuelve el problema de optimización utilizando descenso por gradiente
        self.x = x0
        self.learn = learning_rate 
        self.minimize = minimize
        if not self.minimize:
            self.coef = 1.0 
        condition = True
        i = 0
        while condition and i < Nmax: 
            w = self.f.grad(self.x)
            xp = self.x + self.coef * self.learn * w
            delta = (xp - self.x).norm()
            condition = delta > tol
            self.x = xp
            i += 1
        return xp    

    def restriccion_igualdad(self, x0, k, coef=0, alpha=0.001, tol=0.00001, ln=0.1, tol_solve=0.001):  
        # Resuelve problemas con restricciones de igualdad utilizando proyección ortogonal
        self.x = x0
        
        def h(x):
            return abs(self.h[coef](vector(x)) - k)
        
        f = funcion(h, len(self.x.x))
        z = optim(f)
        self.x = z.solve(self.x, learning_rate=ln, tol=tol_solve) 

        def i(x):
            return self.h[coef](vector(x)) - k
        
        j = funcion(i, len(self.x.x))
        
        condition = True
        while condition:
            gf = self.f.grad(self.x)
            w = gf
            gg = j.grad(self.x)
            v = w.orth(gg)
            xp = self.x + self.coef * alpha * v
            delta = (xp - self.x).norm()
            condition = delta > tol
            self.x = xp
        return self.x, self.f(self.x)

    def restriccion_desigualdad(self, x0, k, alpha=0.001, tol=0.000001, tol2=0.1, ln=0.1, tol_solve=0.001): 
        # Resuelve problemas con restricciones de desigualdad utilizando descenso por gradiente
        self.x = x0
        
        def h(x):
            suma = 0
            for i in range(len(self.g)):
                suma += abs(min((self.g[i](vector(x)) - k), 0))
            return suma
        
        f = funcion(h, len(self.x.x))
        z = optim(f)
        self.x = z.solve(self.x, learning_rate=ln, tol=tol_solve) 
        
        condition = True
        while condition:
            gf = self.f.grad(self.x)
            w = self.coef * gf
            gg = -1 * f.grad(self.x)
            cond = sum(gg.x)
            if cond != 0.0 and (w.inner(gg) < tol2 and abs(h(self.x.x)) < tol2): 
                gg_versor = gg.versor()
                pi = w.inner(gg_versor)
                v = w - pi * gg_versor
                xp = self.x + alpha * v
                delta = (xp - self.x).norm()
                condition = delta > tol
                self.x = xp
            else:
                xp = self.x + alpha * w
                delta = (xp - self.x).norm()
                condition = delta > tol
                self.x = xp 
        return self.x, self.f(self.x)
  
    def restriccion_igualdad_y_desigualdad(self, x0, P, alpha=0.001, tol=0.000001, ln=0.1, tol_solve=0.001): 
        # Resuelve problemas con restricciones de igualdad y desigualdad utilizando penalización
        self.x = x0

        def i(x):
            suma1 = 0
            for i in range(len(self.h)):
                suma1 += (self.h[i](vector(x))) ** 2
            suma2 = 0
            for i in range(len(self.g)):   
                suma2 += (min((self.g[i](vector(x))), 0)) ** 2
            return self.f(vector(x)) + (P * (suma1 + suma2))
        
        f = funcion(i, len(self.x.x))
        z = optim(f)
        self.x = z.solve(self.x, learning_rate=ln, tol=tol_solve)
        
        return self.x, self.f(self.x)
