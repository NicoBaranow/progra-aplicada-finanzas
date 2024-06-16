import matplotlib.pyplot as plt
import sys

sys.path.append('TP2 - Polinomios')
sys.path.append('TP4 - Matrices')

from matrices import matrix
from polis import poly

class Gauss_Seidel(matrix):
    def __init__ (self, v0 = (1,1), F = None, G = None):
        '''
        v0: valor de inicializacion. Inicializado en (1,1) por defecto. 
        F: matriz que define las funciones polinomicas a utilizar en la actualizacion de x
        G: matriz que define las funciones polinomicas a utilizar en la actualizacion de y
        '''
        self.x, self.y = v0
        self.F = F
        self.G = G
        self.history = []
        self.f = self.f_func()
        self.g = self.g_func()

    def f_func(self):
        '''
        Devuelve un polinomio f(x) con y constante
        '''
        coefs = [sum(self.F.get_elem(i+1, j+1)*(self.y**j) for j in range(self.F.c)) for i in range(self.F.r)]
        return poly(self.F.r-1, coefs)

    def g_func(self):
        '''
        Devuelve un polinomio g(y) con x constante
        '''
        coefs = [sum(self.G.get_elem(j+1, i+1)*(self.x**j) for j in range(self.G.r)) for i in range(self.G.c)]
        return poly(self.G.c-1, coefs)
            
    def solve(self, guess=(1,1), tol=0.0001, max_iter=100000, alpha = 0.1):
        '''
        guess: valor de inicializacion. Inicializado en (1,1) por defecto.
        tol: tolerancia para la convergencia. Inicializado en 0.0001 por defecto.
        max_iter: cantidad maxima de iteraciones. Inicializado en 100000 por defecto.
        alpha: Inicializado en 0.1 por defecto.
        '''
        x1, y1 = guess
        iterations = 0
        
        self.history.append((self.x,self.y))

        while iterations < max_iter:
            f = self.f_func()
            g = self.g_func()
            
            x1 = round(self.x - alpha * f(self.x) / f.fprime(1, self.x), 5)
            self.history.append((x1,y1))

            y1 = round(self.y - alpha * g(self.y) / g.fprime(1, self.y), 5)
            self.history.append((x1,y1))
            
            if abs(x1 - self.x) < tol and abs(y1 - self.y) < tol: return self.x, self.y
            
            iterations += 1
            self.x, self.y = x1, y1

        print(f"No se ha encontrado convergencia luego de {max_iter} iteraciones")
        return None, None
    
    def plot_sol(self):
        plt.plot([x for x, y in self.history], [y for x, y in self.history], 'o-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Convergencia del método de Gauss-Seidel')
        plt.show()

F = matrix([-1,0,0,1], 2, 2)

G = matrix([-10,2,5,0], 2, 2)

gs = Gauss_Seidel((1,1), F, G)

print(f"\nPara los polinomios: \n", gs.f.__str__('f(x) = ', 'x'),'\n', gs.g.__str__('g(y) = ', 'y'))
print("\nEl punto de convergencia es:", gs.solve(),'\n') 
gs.plot_sol()
gs = Gauss_Seidel((1,1), G, F)

print(f"\nPara los polinomios: \n", gs.f.__str__('f(x) = ', 'x'),'\n', gs.g.__str__('g(y) = ', 'y'))
print("\nEl punto de convergencia es:", gs.solve(),'\n') 

