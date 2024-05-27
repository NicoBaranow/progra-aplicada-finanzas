import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('TP2 - Polinomios')
sys.path.append('TP4 - Matrices')

from matrices import matrix
from polis import poly

class cub_interp(matrix):
    def __init__ (self, points):
        '''
        points: lista de tuplas (x, y) con los puntos a interpolar
        '''
        self.points = sorted(points)
        self.N = len(points) - 1        

        self.matrix, self.matrix_results = self.build_matrix()
        self.solution_matrix, self.solution_polis = self.solve()

    def build_matrix(self):
        '''
        Devuelve una matriz de splines cúbicos y un vector de resultados para los puntos dados
        '''
        polinomios_elems, polinomios_results = self.polinomios()
        derivada_elems, derivada_results = self.derivada()
        border_elems, border_results = self.border()

        elems = polinomios_elems + derivada_elems + border_elems
        results = polinomios_results + derivada_results + border_results

        return matrix(elems, 4 * self.N, 4 * self.N), matrix(results, 4 * self.N, 1)

    def polinomios (self): 
        '''
        Funcion auxiliar para la construccion de splines cubicos
        '''
        elems = []
        results = []
        for i in range(self.N):
            for j in range(self.N):
                if j == i:
                    x = self.points[i][0]
                    elems.extend([x**3, x**2, x, 1])
                    results.append(self.points[i][1])
                else:
                    elems += [0]*4
            
            for j in range(self.N):
                if j == i:
                    x = self.points[i+1][0]
                    elems.extend([x**3, x**2, x, 1])                    
                    results.append(self.points[i+1][1])
                else:
                    elems += [0]*4

        return elems, results
 
    def derivada (self):
        '''
        Funcion auxiliar para la construccion de splines cubicos
        '''
        elems = []
        results = []
        # Primera derivada
        for i in range(1, self.N):
            for j in range(1, self.N):
                if j == i:
                    x = self.points[i][0]
                    elems.extend([3*x**2, 2*x, 1, 0, -3*x**2, -2*x, -1, 0])
                    results.append(0)
                else:
                    elems += [0]*4
        
        # Segunda derivada
        for i in range(1, self.N):
            for j in range(1, self.N):
                if j == i:
                    x = self.points[i][0]
                    elems.extend([6*x, 2, 0, 0, -6*x, -2, 0, 0])
                    results.append(0)
                else:
                    elems += [0]*4
        
        return elems, results

    def border (self):
        '''
        Funcion auxiliar para la construccion de splines cubicos
        '''
        elems = []
        results = []
        
        # Condiciones de borde
        x = self.points[0][0]
        elems.extend([6*x, 2, 0, 0] + [0]*4*(self.N-1))
        x = self.points[-1][0]
        elems.extend([0]*4*(self.N-1) + [6*x, 2, 0, 0])
        results.extend([0, 0])

        return elems, results

    def solve (self):
        solution = (self.matrix.Minverse() * self.matrix_results)
        
        coeficientes = [[round(x, 4) for x in solution.elems]]
        
        polinomios = []

        for i in range(self.N):
            coefs = coeficientes[0][i * 4 : (i + 1) * 4]
            polinomios.append(poly(3, coefs))
        
        return solution, polinomios

    def interplot(self, xmin, xmax, n=100):
        """
        Grafica la interpolación cúbica en el rango [xmin, xmax] con n puntos.
        """
        x_values = np.linspace(xmin, xmax, n)
        y_values = [self(x) for x in x_values]  

        plt.plot(x_values, y_values, label='Interpolación cúbica')
        
        orig_x = [point[0] for point in self.points]
        orig_y = [point[1] for point in self.points]
        plt.scatter(orig_x, orig_y, color='red', label='Puntos originales')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Interpolación cúbica')
        plt.legend()
        plt.grid(True)
        plt.show()

a = cub_interp([(1, 2.718), (8, 1.118), (9, 1.221)] )

print(a.matrix)
# a.interplot(0, 10)
print(a.matrix_results)
print(a.solution_matrix)


