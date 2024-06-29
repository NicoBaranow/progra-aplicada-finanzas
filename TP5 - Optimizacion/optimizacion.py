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
        '''
        self.f = f
        self.g = opt2d(g) if g is not None else None
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
    
    def grad_call(self, x = (0,0)):
        '''
        Dado un punto (x,y), devuelve el gradiente de f en ese punto.
        En caso de ser un numero imaginaro, se devuelve solo la parte real
        '''
        # Obtener gradiente en el punto dado
        grad_x, grad_y = self.gradf(x)
        
        # Convertir a real si es necesario
        grad_x = grad_x.real if isinstance(grad_x, complex) else grad_x
        grad_y = grad_y.real if isinstance(grad_y, complex) else grad_y
        
        return (grad_x, grad_y)
    
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

    def contour2(self, x0=2, y0=2, repetitions=100000, alpha=0.01):
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
        Devuelve una instancia de la clase matriz correspondiente a la matriz hessiana de f en el punto x
        
        Parametros:
            x: punto a evaluar la matriz hessiana
        
        Retorna:
            matrix: instancia de la clase matriz, con los elementos de f en el punto x
        '''
        return matrix([self.fxx(x), self.fxy(x), self.fxy(x), self.fyy(x)])

    def zona_admisible (self, function, x0 = -5, y0 = -5, learning_rate = 0.0001, max_iter = 100000):
        '''
        Encuentra el punto en la zona admisible más cercano al punto inicial (x0, y0)
        
        Parámetros:
            x0, y0: punto inicial
            tol: tolerancia para la convergencia
            max_iter: número máximo de iteraciones
        
        Retorna:
            tuple: punto en la zona admisible más cercano al punto inicial
        '''

        gradf = function.grad_call((x0, y0))
        xs = [x0]
        ys = [y0]

        iteration = 0
        in_zona = function.f((x0, y0)) >= 0

        while iteration < max_iter and not in_zona:
            x1 = x0 - learning_rate * gradf[0]
            y1 = y0 - learning_rate * gradf[1]

            xs.append(x1)
            ys.append(y1)

            if function.f((x1, y1)) >= 0:
                break  # Encuentra el primer punto en la zona admisible y termina

            # Actualizar gradiente para la siguiente iteración
            gradf = function.grad_call((x1, y1))

            x0, y0 = x1, y1
            iteration += 1
        
        return xs[-1], ys[-1]

    def get_minimum_desigualdad(self, x0, y0, tol = 0.001, epsilon = 0.001, delta = 0.001, max_iter = 100000):
        '''
        Retorna el mínimo de f sujeto a la restricción g(x, y) >= 0
        Parametros:
            x0, y0: punto inicial
            tol: tolerancia para la convergencia
            epsilon: tolerancia para la restricción
            delta: tamaño del paso
            max_iter: número máximo de iteraciones
        Retorna:
            tuple: punto mínimo encontrado
        '''
        
        # Definir una función cuadrática de contorno
        def f_contour_square(x):
            return (self.g([x[0], x[1]]))**2

        # Crear una instancia de opt2d con la función cuadrática de contorno
        self.f_contour = opt2d(f_contour_square)

        # Encontrar la zona admisible
        XS, ys = self.zona_admisible(self.f_contour, x0, y0)
        x0, y0 = XS, ys
        print(self.gradf((x0, y0)))
        # Inicializar los gradientes
        gradf = vector([self.g.gradf([x0, y0])[0], self.g.gradf([x0, y0])[1]])
        gradg = vector([self.g.gradf([x0, y0])[0], self.g.gradf([x0, y0])[1]])

        xs, ys = [], []
        iteration = 0

        while iteration < max_iter:
            if self.g.f([x0, y0]) > epsilon:
                x1, y1 = x0 - delta * gradf.x[0], y0 - delta * gradf.x[1]
                xs.append(x1)
                ys.append(y1)
                in_zona = True
            else:
                if gradg.inner(gradf * -1) < 0:
                    v = (-gradf.x[0] - (gradg.inner(gradf * -1) * gradg.versor()).x[0],
                         -gradf.x[1] - (gradg.inner(gradf * -1) * gradg.versor()).x[1])
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                    xs.append(x1)
                    ys.append(y1)
                    in_zona = False
                else:
                    x1, y1 = x0 - delta * gradf.x[0], y0 - delta * gradf.x[1]
                    xs.append(x1)
                    ys.append(y1)
                    in_zona = True

            # Actualizar los gradientes
            gradf.x = [self.grad_call([x1, y1])[0], self.grad_call([x1, y1])[1]]
            gradg.x = [self.g.grad_call([x1, y1])[0], self.g.grad_call([x1, y1])[1]]

            # Comprobar la colinealidad
            col = gradf.versor().inner(gradg.versor())
            if not in_zona and abs(col - 1) < tol:
                break
            if in_zona and abs(gradf.x[0]) < epsilon and abs(gradf.x[1]) < epsilon:
                break

            x0, y0 = x1, y1
            iteration += 1

        return xs[-1], ys[-1]



class opt2d_restriccion_igualdad(opt2d):
    def __init__(self, f, g):
        '''
        f: función de dos variables
        g: función de dos variables que representa la restriccion
        '''
        super().__init__(f)
        self.g = opt2d(g) #instanciamos un objeto de la clase opt2d con la función g

    def gradg(self, x = (0,0)):
        '''
        Dado un punto (x,y), devuelve el gradiente de g en ese punto
        '''
        return self.g.gradf(x)

    def campo_gradiente(self, x_range = (-50,50), y_range = (-50,50), nx = 100, ny = 100):
        '''
        Devuelve los parámetros necesarios para hacer un plt.quiver y graficar el campo gradiente de la restriccion g en un rango dado
        Parametros:
            x_range: tupla con los valores de x inicial y final
            y_range: tupla con los valores de y inicial y final
            nx: cantidad de puntos en x
            ny: cantidad de puntos en y

        Retorna:
            X, Y, U, V: listas con los valores necesarios para graficar el campo gradiente con plt.quiver()
        '''
        return self.g.campo_gradiente(x_range, y_range, nx, ny)

    def get_minimum(self, x0, y0, tol=0.001, delta=0.01, max_iter=100000):
        '''
        
        '''
        # Define g^2 de la restricción
        def f_contour_square(x):
            return abs(self.g.f((x[0], x[1])))  # Evalúa la restricción g (instancia de opt2d) en el punto (x, y)
        
        # Encuentra el mínimo de g^2 en (x0, y0)
        self.f_contour = opt2d(f_contour_square)  # Instanciamos un objeto de la clase opt2d con la restricción g^2
        point, iters = self.f_contour.gdescent(x0, y0)
        xs, ys = [point[0]], [point[1]]
        x0, y0 = xs[-1], ys[-1]
        
        # Algoritmo de contour2
        gradf = vector(self.grad_call((x0, y0)))
        gradg = vector(self.g.grad_call((x0, y0)))
        iteration = 0
        
        while iteration < max_iter:
            try:
                v = vector([-gradf.x[0] - (gradg.versor().inner(gradf) * gradg.versor()).x[0],
                            -gradf.x[1] - (gradg.versor().inner(gradf) * gradg.versor()).x[1]])
            except ValueError:
                print("Encountered zero vector during normalization, stopping iteration.")
                break
            
            x1, y1 = x0 + delta * v.x[0], y0 + delta * v.x[1]
            
            xs.append(x1)
            ys.append(y1)
            
            gradf = vector(self.grad_call((x1, y1)))
            gradg = vector(self.g.grad_call((x1, y1)))

            col = gradf.versor().inner(gradg.versor())
            if abs(col - 1) < tol:
                break
            
            x0, y0 = x1, y1
            iteration += 1
    
        return xs[-1], ys[-1]

class opt2d_restriccion_desigualdad(opt2d_restriccion_igualdad):
    def __init__ (self, f, g):
        super().__init__(f, g)
        self.f = opt2d(f)

    


def objetivo(x):
    return (x[0]+5+x[1]*4)**2

def restriccion(x):
    return x[1]+x[0]+2

a = opt2d(objetivo, restriccion)
print(a.get_minimum_desigualdad(6,6))

#prints(-15, -15)


# resultado, iteraciones, historia = optimizador.optimize_with_constraint(punto_inicial)
# print(f"El mínimo encontrado con restriccion es {resultado} después de {iteraciones} iteraciones.")

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
