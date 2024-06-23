import sys
import matplotlib.pyplot as plt
sys.path.append('TP4 - matrices')
sys.path.append('TP6 - Vectores')
from matrices import matrix
from vectores import vector
import numpy as np

def my_linspace(start, stop, num=100):
    """
    Genera `num` puntos igualmente espaciados entre `start` y `stop`, incluyendo ambos extremos.
    
    Parámetros:
        start (float): El valor inicial de la secuencia.
        stop (float): El valor final de la secuencia.
        num (int): Número de puntos a generar.
        
    Retorna:
        list: Una lista de `num` valores igualmente espaciados.
    """
    if num <= 0:
        return []
    
    step = (stop - start) / (num - 1)
    
    return [start + step * i for i in range(num)]

class opt2d(matrix):
    def __init__(self, f, g = None):
        '''
        f: función de dos variables
        g: función de dos variables que representa la restricción
        '''
        self.f = f
        self.g = g
        self.hx = 0.0001
        self.hy = 0.0001

    def fx(self, x = (0,0)):
        return (self.f((x[0] + self.hx, x[1])) - self.f((x[0] - self.hx, x[1]))) / (2 * self.hx)

    def fy(self, x = (0,0)):
        return (self.f((x[0], x[1] + self.hy)) - self.f((x[0], x[1] - self.hy))) / (2 * self.hy)

    def fxx(self, x = (0,0)):
        return (self.fx((x[0] + self.hx, x[1])) - self.fx((x[0] - self.hx, x[1]))) / (2 * self.hx)

    def fyy(self, x = (0,0)):
        return (self.fy((x[0], x[1] + self.hy)) - self.fy((x[0], x[1] - self.hy))) / (2 * self.hy)

    def fxy(self, x = (0,0)):
        return (self.fx((x[0], x[1] + self.hy)) - self.fx((x[0], x[1] - self.hy))) / (2 * self.hy)

    def gradf(self, x = (0,0)):
        '''
        Dado un punto (x,y), devuelve el gradiente de f en ese punto
        '''
        return (self.fx(x), self.fy(x))
    
    def fv(self, x = (0, 0), v = (2, 3)):
        '''
        Dado un punto (x,y) y un vector v, devuelve la derivada direccional de f en esa dirección
        '''

        grad = self.gradf(x)
        norma = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return grad[0]*v[0]/norma + grad[1]*v[1]/norma

    def campo_gradiente(self, x_range = (-50,50), y_range = (-50,50), nx = 100, ny = 100):
        '''
        Devuelve los parámetros necesarios para hacer un plt.quiver y graficar el campo gradiente de f en un rango dado
        Parametros:
        x_range: tupla con los valores de x inicial y final
        y_range: tupla con los valores de y inicial y final
        nx: cantidad de puntos en x
        ny: cantidad de puntos en y

        Retorna:
            X, Y, U, V: listas con los valores necesarios para graficar el campo gradiente con plt.quiver()
        '''
        x = my_linspace(x_range[0], x_range[1], nx)
        y = my_linspace(y_range[0], y_range[1], ny)
        
        X = [[x_i for x_i in x] for _ in y]
        Y = [[y_j for _ in x] for y_j in y]
        U = [[0 for _ in range(nx)] for _ in range(ny)]
        V = [[0 for _ in range(nx)] for _ in range(ny)]

        for i in range(ny):
            for j in range(nx):
                grad = self.gradf((X[i][j], Y[i][j]))
                U[i][j] = grad[0]
                V[i][j] = grad[1]

        return X, Y, U, V

    def contour2(self, x0=2, y0=2, repetitions=1000000, alpha=0.01):
        '''
        Devuelve las coordenadas necesarias para graficar las curvas de nivel de f que pasan por el punto (x0, y0) en forma de dos listas

        Parametros:
            x0, y0: punto inicial
            repetitions: cantidad de puntos a graficar
            alpha: tamaño del paso

        Retorna:
            x_coords1, y_coords1: listas con las coordenadas necesarias para graficar las curvas de nivel
        '''
        
        # Convertimos x0 e y0 a una instancia de matrix
        x0 = matrix([x0, y0], 2, 1)
        grad = self.gradf((x0.elems[0], x0.elems[1]))
        grad = vector(grad)  # Convertimos el gradiente a una instancia de vector
        grad_matrix = matrix(grad.x, 2, 1)  # Convertimos el vector a matrix
        
        # Matriz de rotación antihoraria de 90 grados
        r1 = matrix([0, -1, 1, 0], 2, 2)

        # Vector perpendicular al gradiente
        v1 = r1 * grad_matrix
        v1 = vector(v1.elems).versor()  # Normalizamos el vector

        xy_list1 = []

        for _ in range(repetitions):
            x1 = x0 + alpha * v1
            xy_list1.append(x1.elems)

            x0 = x1
            grad = self.gradf((x0.elems[0], x0.elems[1]))
            grad = vector(grad)
            grad_matrix = matrix(grad.x, 2, 1)  # Convertimos el vector a matrix

            v1 = r1 * grad_matrix
            v1 = vector(v1.elems).versor()  # Normalizamos el vector

        x_coords1 = [x[0] for x in xy_list1]
        y_coords1 = [y[1] for y in xy_list1]

        return x_coords1, y_coords1

    def gdescent(self, x0 = 2, y0 = 2, learning_rate=0.1, tol=0.0000001, max_iter=100000):
        """
        Realiza el descenso por gradiente para encontrar un mínimo.

        Parámetros:
            x0, y0 (float): Punto inicial.
            learning_rate (float): tasa de aprendizaje. Multiplica al gradiente.
            tol (float): Tolerancia para la convergencia.
            max_iter (int): Número máximo de iteraciones.
            
        Retorna:
            tuple: Punto extremo encontrado.
            int: Número de iteraciones realizadas.
        """
        
        for n in range(max_iter):
            grad = self.gradf((x0, y0))
            xn_next = (x0 - learning_rate * grad[0], y0 - learning_rate * grad[1])

            if ((xn_next[0] - x0) ** 2 + (xn_next[1] - y0) ** 2) ** 0.5 < tol:
                return (xn_next[0], xn_next[1]), n + 1

            x0, y0 = xn_next

        print("El método de descenso por gradiente no convergió después de", max_iter, "iteraciones.")

        return (x0, y0), max_iter
    
    def gascent (self, x0 = 2, y0 = 2, learning_rate=0.1, tol=0.0000001, max_iter=100000):
        """
        Realiza el descenso por gradiente a la inversa para encontrar un maximo.
        
        Parámetros:
            x0, y0 (float): Punto inicial.
            learning_rate (float): tasa de aprendizaje. Multiplica al gradiente.
            tol (float): Tolerancia para la convergencia.
            max_iter (int): Número máximo de iteraciones.
            
        Retorna:
            tuple: Punto extremo encontrado.
            int: Número de iteraciones realizadas.
        """
        
        for n in range(max_iter):
            grad = self.gradf((x0, y0))
            xn_next = (x0 + learning_rate * grad[0], y0 + learning_rate * grad[1])

            if ((xn_next[0] - x0) ** 2 + (xn_next[1] - y0) ** 2) ** 0.5 < tol:
                return (xn_next[0], xn_next[1]), n + 1

            x0, y0 = xn_next

        print("El método de ascenso por gradiente no convergió después de", max_iter, "iteraciones.")
        return (x0, y0), max_iter
    
    def polytopes(self, initial_points, tol=1e-7, max_iter=100000):
        """
        Implementa el método de Polytopes para encontrar un mínimo.

        Parámetros:
            initial_points (list): Lista de puntos iniciales en (x, y).
            tol (float): Tolerancia para la convergencia.
            max_iter (int): Número máximo de iteraciones.
            
        Retorna:
            tuple: Punto mínimo encontrado.
            int: Número de iteraciones realizadas.
            list: Lista de puntos generados para graficar.
        """
        points = initial_points
        z_values = [self.combined_function(p) for p in points]
        history = []

        for n in range(max_iter):
            points, z_values = zip(*sorted(zip(points, z_values), key=lambda pair: pair[1]))
            points, z_values = list(points), list(z_values)
            history.append(points.copy())

            centroid = [(sum(p[i] for p in points[:-1]) / (len(points) - 1)) for i in range(2)]
            worst_point = points[-1]
            reflected = [centroid[i] + (centroid[i] - worst_point[i]) for i in range(2)]

            if self.combined_function(reflected) < z_values[-2]:
                points[-1] = reflected
                z_values[-1] = self.combined_function(reflected)
                if self.combined_function(reflected) < z_values[0]:
                    expanded = [centroid[i] + 2 * (centroid[i] - worst_point[i]) for i in range(2)]
                    if self.combined_function(expanded) < self.combined_function(reflected):
                        points[-1] = expanded
                        z_values[-1] = self.combined_function(expanded)
                    else:
                        points[-1] = reflected
                        z_values[-1] = self.combined_function(reflected)
            else:
                if self.combined_function(reflected) < z_values[-1]:
                    points[-1] = reflected
                    z_values[-1] = self.combined_function(reflected)
                else:
                    contracted = [centroid[i] + 0.5 * (worst_point[i] - centroid[i]) for i in range(2)]
                    if self.combined_function(contracted) < z_values[-1]:
                        points[-1] = contracted
                        z_values[-1] = self.combined_function(contracted)
                    else:
                        for i in range(1, len(points)):
                            points[i] = [points[0][j] + 0.5 * (points[i][j] - points[0][j]) for j in range(2)]
                        z_values = [self.combined_function(p) for p in points]

            if max(abs(z - z_values[0]) for z in z_values) < tol:
                return points[0], n + 1, history

        print("El método de Polytopes no convergió después de", max_iter, "iteraciones.")
        return points[0], max_iter, history

    def hessiano (self, x = (0,0)):
        '''
        Devuelve el hessiano de f en el punto x
        
        Parametros:
            x: punto a evaluar la matriz hessiana
        
        Retorna:
            matrix: instancia de la clase matriz, con los elementos de f en el punto x
        '''
        return matrix([self.fxx(x), self.fxy(x), self.fxy(x), self.fyy(x)])

    def combined_function(self, x):
        '''
        Combina la función objetivo y la restricción elevándolas al cuadrado y sumándolas
        '''
        if self.g:
            return self.f(x)**2 + 10000 * self.g(x)**2
        else:
            return self.f(x)


    def optimize_with_constraint(self, initial_point, tol=1e-7, max_iter=100000):
        '''
        Optimiza la función objetivo bajo la restricción utilizando el método de Polytopes
        
        Parámetros:
            initial_point: punto inicial para la optimización
            tol: tolerancia para la convergencia
            max_iter: número máximo de iteraciones

        Retorna:
            result: punto óptimo encontrado
            iterations: número de iteraciones realizadas
            history: historial de puntos generados para graficar
        '''

        initial_points = [
            (initial_point[0] + 0.1, initial_point[1]),
            (initial_point[0], initial_point[1] + 0.1),
            (initial_point[0] + 0.1, initial_point[1] + 0.1)
        ]
        
        result, iterations, history = self.polytopes(initial_points, tol, max_iter)
        return result, iterations, history

def objetivo(x):
    return x[1]**2 + x[0]**2

def restriccion(x):
    return x[1]+10


optimizador = opt2d(f=restriccion, g = objetivo)
punto_inicial = (0, -10)


resultado, iteraciones, historia = optimizador.optimize_with_constraint(punto_inicial)
print(f"El mínimo encontrado con restriccion es {resultado} después de {iteraciones} iteraciones.")

# resultado2, iteraciones2 = optimizador.gdescent(x0=2, y0=3, learning_rate=0.1, tol=0.0001, max_iter=100000)
# print(f"El mínimo encontrado con descenso por gradiente es {resultado2} después de {iteraciones2} iteraciones.")

# resultado3, iteraciones3, history = optimizador.polytopes([(5, -5), (10, 5), (-5, -2.5)])
# print(f"El mínimo encontrado con polytopes es {resultado3} después de {iteraciones3} iteraciones.")
# opt = opt2d(function, restriction_function) #le pasamos una función como parametro, pero no la ejecutamos, por eso sin ()
# x_opt, y_opt = opt.minimize_with_constraints(0.5, 0.5)
# print(f"Minimize with constraints result: x={x_opt}, y={y_opt}")

# # Encontrar mínimo usando descenso por gradiente
# min_point, min_iters = opt.gdescent(x0=2, y0=0, learning_rate=0.1, tol=0.0001, max_iter=100000)
# print(f"El mínimo encontrado es {min_point} después de {min_iters} iteraciones.")

# # Encontrar máximo usando ascenso por gradiente
# max_point, max_iters = opt.gascent(x0=2, y0=0, learning_rate=0.1, tol=0.0001, max_iter=100000)
# print(f"El máximo encontrado es {max_point} después de {max_iters} iteraciones.")

# # Encontrar mínimo usando método de Polytopes
# puntos_iniciales = [(5, -5), (10, 5), (-5, -2.5)]
# poly_point, poly_iters, history = opt.polytopes(puntos_iniciales)
# print(f"El mínimo encontrado con polytopes es {poly_point} después de {poly_iters} iteraciones. F en ese punto es: {opt.f(poly_point)}")

# #Comprobación de puntos máximos y mínimos 
# print(f'El gradiente en el punto {min_point} es {opt.gradf(min_point)}')
# print(f'El gradiente en el punto {max_point} es {opt.gradf(max_point)}')
# print(f'El gradiente en el punto {poly_point} es {opt.gradf(poly_point)}')

# ######Graficar curvas de nivel, campo gradiente y extremos######

# # Encontrar curvas de nivel y campo gradiente
# plt.figure(figsize=(8, 8))
# x_coords1, y_coords1 = opt.contour2(x0=1, y0=1, repetitions=22500, alpha=0.001)
# x_coords2, y_coords2 = opt_restricted.contour2(x0=1, y0=2, repetitions=3500, alpha=0.001)

# plt.plot(x_coords1, y_coords1, label='Curva de nivel')
# plt.plot(x_coords2, y_coords2, label='Curva de nivel restringida')

# x,y,u,v = opt.campo_gradiente(x_range=(-100,100), y_range=(-100,100), nx=20, ny=20)
# x1, y1, u1, v1 = opt_restricted.campo_gradiente(x_range=(-10,10), y_range=(-10,10), nx=20, ny=20)

# opt.campo_gradiente()
# plt.quiver(x, y, u, v, label='Campo gradiente funcion objetivo')
# plt.quiver(x1, y1, u1, v1, label='Campo gradiente funcion restringida', color='red')

# plt.scatter(point[0], point[1], color='green', label='Mínimo encontrado')
# plt.scatter(point2[0], point2[1], color='red', label='Máximo encontrado')

# for iteration, points in enumerate(history):
#     x_coords = [p[0] for p in points]
#     y_coords = [p[1] for p in points]
#     plt.scatter(x_coords, y_coords)
#     plt.plot(x_coords, y_coords)

# # # plt.scatter(poly_point[0], poly_point[1], color='purple', label='Mínimo con Polytopes')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Curvas de nivel, campo gradiente y extremos')
# plt.grid(True)
# plt.legend()
# plt.show()