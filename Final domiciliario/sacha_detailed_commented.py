import matplotlib.pyplot as plt
import math
import numpy as np
import time
import sys

sys.path.append('TP4 - matrices')
sys.path.append('TP6 - Vectores')
from matrices_suggested import lineq, myarray
from vectores import vector

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

class opt2d:
    
    def __init__(self, func, hx=0.001, hy=0.001):
        self.f = func
        self.hx = hx
        self.hy = hy
        
    def fx(self, x, y):
        # derivada partial respeto a x
        return (self.f(x + self.hx, y) - self.f(x - self.hx, y)) / (2 * self.hx)

    def fy(self, x, y):
        # derivada partial respeto a y
        return (self.f(x, y + self.hy) - self.f(x, y - self.hy)) / (2 * self.hy)

    def fxx(self, x, y):
        # 2nda derivada partial respeto a x
        return (self.f(x + self.hx, y) - 2*self.f(x, y) + self.f(x - self.hx, y)) / (self.hx**2)

    def fxy(self, x, y):
        # derivada cruzada
        return (self.f(x + self.hx, y + self.hy) - self.f(x - self.hx, y + self.hy) - self.f(x + self.hx, y - self.hy) + self.f(x - self.hx, y - self.hy)) / (4 * self.hx * self.hy)

    def fyy(self, x, y):
        # 2nda derivada partial respeto a y
        return (self.f(x, y + self.hy) - 2*self.f(x, y) + self.f(x, y - self.hy)) / (self.hy**2)

    def gradf(self):
        return (self.fx, self.fy)
    
    def grad_call(self, x, y):
        # Converting complex to real if necessary
        grad_x = self.gradf()[0](x, y)
        grad_y = self.gradf()[1](x, y)
        return [grad_x.real if isinstance(grad_x, complex) else grad_x,
                grad_y.real if isinstance(grad_y, complex) else grad_y]
    
    def fv(self, punto, v):
        x,y = punto
        vx, vy = v
        return self.gradf()[0](x,y) * vx + self.gradf()[1](x,y) * vy
    
    def campo_gradiente(self, xmin, xmax, ymin, ymax, nx, ny, points=None, ax=None, fig=None):
        # Crear los linespaces
        x = [xmin + i * (xmax - xmin) / (nx - 1) for i in range(nx)]
        y = [ymin + i * (ymax - ymin) / (ny - 1) for i in range(ny)]
        
        # Crear una grid de puntos
        X, Y = [[0] * nx for _ in range(ny)], [[0] * nx for _ in range(ny)]
        for i in range(ny):
            for j in range(nx):
                X[i][j] = x[j]
                Y[i][j] = y[i]

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        
        # Calcular la norma maxima
        max_norm = 0
        gradients = []
        for i in range(ny):
            for j in range(nx):

                dx, dy = self.grad_call(X[i][j], Y[i][j])
                norm = math.sqrt(dx**2 + dy**2)
                gradients.append((X[i][j], Y[i][j], dx, dy, norm))
                if norm > max_norm:
                    max_norm = norm
        
        # Dibuja los gradientes
        for x, y, dx, dy, norm in gradients:
            if max_norm > 0:
                dx_normalized, dy_normalized = (dx / max_norm, dy / max_norm)
            else:
                dx_normalized, dy_normalized = (dx, dy)
            ax.arrow(x, y, dx_normalized, dy_normalized, head_width=0.2, head_length=0.4, fc='black', ec='black')
        
        if points:
            xs, ys = points
            ax.plot(xs, ys, 'ro-', markersize=1, linewidth=1)  

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        if fig is None and ax is None:
            plt.show()

    def contour2(self, x0, y0, xmin, xmax, ax = None, fig = None, color = "ro-", learning = 0.001, rep = 10000):
        
        puntos_ah = []  # Puntos para rotación antihoraria
        puntos_h = []   # Puntos para rotación horaria
    
        # Rotación antihoraria de 90 grados
        for _ in range(rep):
            grad = (self.gradf()[0](x0, y0), self.gradf()[1](x0, y0))
            v = (-grad[1], grad[0])
            x1 = (x0 + learning*v[0], y0 + learning*v[1])
            x0, y0 = x1
            puntos_ah.append(x1)
    
        # Rotación horaria de 90 grados
        x0, y0 = puntos_ah[0]  # Reset to starting point
        for _ in range(rep):
            grad = (self.gradf()[0](x0, y0), self.gradf()[1](x0, y0))
            v = (grad[1], -grad[0])
            x1 = (x0 + learning*v[0], y0 + learning*v[1])
            x0, y0 = x1
            puntos_h.append(x1)
    
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        xs_ah, ys_ah = zip(*puntos_ah)
        xs_h, ys_h = zip(*puntos_h)
    
        ax.plot(xs_ah, ys_ah, color, markersize=1, linewidth=1)
        ax.plot(xs_h, ys_h, color, markersize=1, linewidth=1)
    
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        if fig is None and ax is None:
            plt.show()
            
    def contour(self, k, xmin = -5, xmax = 5, ymin = -5, ymax=5, ax = None, fig = None, color = "ro-", learning = 0.001, rep = 10000, x_prop = None, y_prop = None, isUtilidad = False):
        for j in [-2,2]:
            for l in [-2,2]: 
                
                def f_contour_square(x,y):
                    return (self.f(x,y) - k)**2

                self.f_contour = opt2d(f_contour_square)
                
                # Si uno de los gdecent arrancados encontra (x,y) tal que f_contour(x,y) = 0, no necesita arrancar otros gdescent
                x0, y0 = self.f_contour.gdescent_global(j, l, xmin=xmin, xmax = xmax, ymin = ymin, ymax =ymax,stopValue=0, isUtilidad= isUtilidad)
                if x_prop is not None and y_prop is not None:
                    x0, y0 = self.f_contour.gdescent_global(x_prop, y_prop, stopValue=0, isUtilidad= isUtilidad)

                puntos_ah = []  # Puntos para rotación antihoraria
                puntos_h = []   # Puntos para rotación horaria

                # Rotación antihoraria de 90 grados
                for _ in range(rep):
                    grad = self.grad_call(x0, y0)
                    v = (-grad[1], grad[0])
                    x1 = (x0 + learning*v[0], y0 + learning*v[1])
                    x0, y0 = x1
                    puntos_ah.append(x1)
            
                # Rotación horaria de 90 grados
                x0, y0 = puntos_ah[0]  # Reset to starting point
                for _ in range(rep):
                    grad = self.grad_call(x0, y0)
                    v = (grad[1], - grad[0])
                    x1 = (x0 + learning*v[0], y0 + learning*v[1])
                    x0, y0 = x1
                    puntos_h.append(x1)
            
                # Affichage des résultats
                if fig is None and ax is None:
                    fig, ax = plt.subplots()
                xs_ah, ys_ah = zip(*puntos_ah)
                xs_h, ys_h = zip(*puntos_h)
            
                ax.plot(xs_ah, ys_ah, color, markersize=1, linewidth=1)
                ax.plot(xs_h, ys_h, color, markersize=1, linewidth=1)
            
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([xmin, xmax])
                ax.set_aspect('equal', adjustable='box')
                plt.grid(True)
        if fig is None and ax is None:
            plt.show()
        
    def contours(self, start = 0, stop = 5, step = 1, xmin=-5, xmax=5, ymin=-5, ymax=5 , learning=0.001, rep=10000, fig = None, ax = None, isUtilidad=False):
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        bleu_couleurs = [
            "#191970",  
            "#000080",  
            "#003153",  
            "#0047AB",  
            "#4169E1",  
            "#4682B4",  
            "#4682B4",  
            "#007FFF",  
            "#ADD8E6",  
            "#7DF9FF"   
        ]

        for idx, i in enumerate(np.arange(start, stop, step)):
            color = bleu_couleurs[idx % len(bleu_couleurs)]  
            self.contour(i, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, learning=learning, rep=rep, ax=ax, fig=fig, color=color,isUtilidad = isUtilidad)
        if fig is None and ax is None:
            plt.show()
        
    def gdescent(self,x0=1,y0=4,delta=0.01,tol=0.00001,Nmax=100000, returnAsPoints = False):

        fprimex, fprimey = self.gradf()
        learningrate = delta
        xs = [x0]
        ys = [y0]
        
        count = 0
        while count < Nmax:

            grad_x = fprimex(x0, y0)
            grad_y = fprimey(x0, y0)
            
            x1 = x0 - learningrate * grad_x
            y1 = y0 - learningrate * grad_y
            

            xs.append(x1)
            ys.append(y1)
            
            norm = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            if norm < tol:
                break
            
            x0, y0 = x1, y1
            count += 1
        
        if not returnAsPoints:        
            return (xs[-1], ys[-1])

        else:
            return xs, ys
    
    def gdescent_global(self, x0=1, y0=1, delta=0.01, tol=0.0001, Nmax=10000, returnAsPoints=False, trials=10, xmin = -5, xmax = 5, ymin = -5, ymax= 5, stopValue = None, isUtilidad = False):
        best_min = None
        best_points = None
        best_value = float('inf')
        for _ in range(trials):

            x_start = np.random.uniform(xmin, xmax)
            y_start = np.random.uniform(xmin, xmax)
            
            # Initialisation
            fprimex, fprimey = self.gradf()

            learningrate = delta
            xs = [x_start]
            ys = [y_start]
            x0, y0 = x_start, y_start
            count = 0
            
            while count < Nmax:
                grad_x = fprimex(x0, y0)
                grad_y = fprimey(x0, y0)
                x1 = x0 - learningrate * grad_x
                y1 = y0 - learningrate * grad_y
                if (x1 < 0 or y1 <0) and isUtilidad:
                    break
                xs.append(x1)
                ys.append(y1)
                
                norm = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                if norm < tol:
                    break
                x0, y0 = x1, y1
                count += 1
            
            current_value = self.f(x0, y0)

            if current_value < best_value:
                best_value = current_value
                best_min = (x0, y0)
                best_points = (xs, ys)
            if stopValue is not None:
                # Si restriccion(x0, y0) < 0.01, es muy probable que esta el minimo que buscamos
                if abs(self.f(x0, y0) - stopValue) < delta: 
                    break
        if not returnAsPoints:
            return best_min
        else:
            return best_points

    def hessienna(self, x, y):
        return myarray([self.fxx(x,y), self.fxy(x, y), self.fxy(x, y), self.fyy(x, y)], 2, 2)

    def minimumState(self, H):
        if H.det() < 0:
            print("Point Col")
        elif H.det() > 0 and (H.elems[2] + H.elems[3]) > 0:
            print("Minimum local")
        else:
            print("Maximum local")
    
class opt2d_constraint(opt2d):
    
    def __init__(self, func, restriccion):
        self.g = opt2d(restriccion)
        super().__init__(func)

    
    # Campo gradiente de f + curva nivel de f + restriccion g
    def plot1(self, ax, fig, xmin, xmax, ymin, ymax, nx, ny, rep = 10000, learning = 0.001, inequality = False, color = "g-", isUtilidad = False):
        # Champ gradient
        self.campo_gradiente(xmin, xmax, ymin, ymax, nx, ny, ax = ax, fig = fig)

        # Courbes de niveau de f
        self.contours(xmin = xmin, xmax = xmax, ymin=ymin, ymax=ymax, ax=ax, fig=fig, isUtilidad=isUtilidad)

        # Affichage de g(x, y) = 0 (donc courbe de niveau de 0)
        self.g.contour(0, xmin, xmax, ax = ax, fig = fig, color = color, rep=rep, learning=learning, isUtilidad=isUtilidad)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        if not inequality:
            plt.show()
        
        
    # Campo gradiente de g + curva nivel de g 
    def plot2(self, ax, fig, xmin, xmax, ymin, ymax, nx, ny, x0, y0, rep = 10000, learning = 0.001):
        
        # Campo gradiente de g
        self.g.campo_gradiente(xmin, xmax, ymin, ymax, nx, ny, ax = ax, fig = fig)
        
        # Curvas de nivel de g
        self.g.contours(xmin = xmin, xmax = xmax, ax=ax, fig=fig)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
        
    # Curvas  de nivel de f + restriccion
    def plot3(self, ax, fig, xmin, xmax, ymin, ymax, nx, ny, x0, y0, rep = 10000, learning = 0.001):
        
        # Curvas de nivel de f
        self.contours(xmin = xmin, xmax = xmax, ax=ax, fig=fig)
        
        # Restriccion g
        self.g.contour(0, xmin= xmin, xmax = xmax, ax=ax, fig=fig)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
        
    # Trayectoria para encontrar el minimum
    def plot4(self, x0, y0, ax, fig, xmin, xmax, ymin, ymax, nx, ny, rep = 10000, learning = 0.001):
        
        # Curvas de nivel de f
        self.contours(xmin = xmin, xmax = xmax, ax=ax, fig=fig)
        
        # Restriction
        self.g.contour(0, xmin = xmin, xmax = xmax, ax=ax, fig=fig, color="g-")
        
        # Plot los puntos de la busqueda del minimum
        points = self.get_minimum(x0, y0, returnAsPoints=True)
        print("Minimum : ", points[0][-1], points[1][-1])
        self.campo_gradiente(xmin, xmax, ymin, ymax, nx=10, ny=10, ax=ax, fig=fig,points =points)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
        
    def get_minimum(self, x0, y0, tol=0.001, delta=0.01, max_iter=100000, returnAsPoints=False):
        
        # Defines g²
        def f_contour_square(x, y):
            return abs(self.g.f(x, y))
        
        # Encontras el minimo de g² (x0, y0)
        self.f_contour = opt2d(f_contour_square)
        xs, ys = self.f_contour.gdescent(x0, y0, returnAsPoints = True)
        x0, y0 = xs[-1], ys[-1]
        
        # Algoritmo de contour2
        gradf = vector([self.grad_call(x0, y0)[0],self.grad_call(x0, y0)[1]])
        gradg = vector([self.g.grad_call(x0, y0)[0],self.g.grad_call(x0, y0)[1]])
        iteration = 0
        
        while iteration < max_iter:
            v = (-gradf.elems[0] - (gradg.versor().inner(gradf.minus()) * gradg.versor()).elems[0],
                 -gradf.elems[1] - (gradg.versor().inner(gradf.minus()) * gradg.versor()).elems[1])
            
            x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
            
            xs.append(x1)
            ys.append(y1)
            
            gradf.elems = [self.grad_call(x1, y1)[0],self.grad_call(x1, y1)[1]]
            gradg.elems = [self.g.grad_call(x1, y1)[0],self.g.grad_call(x1, y1)[1]]

            
            #Colinéaire ?
            col = gradf.versor().inner(gradg.versor())
            if abs(col - 1) < tol:
                break
            
            x0, y0 = x1, y1
            iteration += 1
  
        if returnAsPoints:
            return xs, ys
        else:
            return xs[-1], ys[-1]

class opt2d_constraint_inequality(opt2d_constraint):
    
    def __init__(self, func, restriccion):
        super().__init__(func, restriccion)
        self.func = opt2d(func)
        
        
    def find_zona_admissible(self, function, x0=1,y0=4,delta=0.0001,tol=0.00001,Nmax=100000, returnAsPoints = False):
         # Récupération des fonctions de dérivées partielles
         
         gradf = function.grad_call(x0, y0)
         learningrate = delta
         xs = [x0]
         ys = [y0]
         
         count = 0
         in_zona = False
         
         # Vérifier les conditions d'arrêt 
         if self.g.f(x0, y0) >= 0:
             in_zona = True
             
         while count < Nmax and not in_zona:
             # Calcul des nouvelles valeurs des variables
             
             x1 = x0 - learningrate * gradf[0]
             y1 = y0 - learningrate * gradf[1]
            
             
             # Ajouter les nouvelles valeurs aux listes
             xs.append(x1)
             ys.append(y1)
             
            
             # Vérifier les conditions d'arrêt 
             if self.g.f(x1, y1) >= 0:
                 break
             

             # Mettre à jour les variables pour la prochaine itération
             x0, y0 = x1, y1
             gradf = function.grad_call(x0, y0)
             count += 1
         
         if not returnAsPoints:        
             return (xs[-1], ys[-1])

         else:
             return xs, ys
        
        
    def get_minimum_2(self, x0, y0, tol=0.001, epsilon=0.001,delta=0.001, max_iter=10000, returnAsPoints=False):
        def f_contour_square(x, y):
            return (self.g.f(x, y))**2
                
        self.f_contour = opt2d(f_contour_square)
        
        xs, ys = self.find_zona_admissible(self.f_contour, x0, y0, returnAsPoints=True)
        x0, y0 = xs[-1], ys[-1]
        gradf = vector([self.grad_call(x0, y0)[0],self.grad_call(x0, y0)[1]])
        gradg = vector([self.g.grad_call(x0, y0)[0],self.g.grad_call(x0, y0)[1]])


        iteration = 0
        while iteration < max_iter:
            if self.g.f(x0, y0)>epsilon:
                x1, y1 = x0 - delta * gradf.elems[0], y0 - delta*gradf.elems[1]
                xs.append(x1)
                ys.append(y1)
                in_zona = True
            else:
                #On est sur la restriction, on veut savoir si gradf tend vers la zone admissible
                if gradg.inner(gradf.minus()) < 0:
                    
                    v = (-gradf.elems[0] - (gradg.inner(gradf.minus()) * gradg.versor()).elems[0],
                         -gradf.elems[1] - (gradg.inner(gradf.minus()) * gradg.versor()).elems[1])
                
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                
                    xs.append(x1)
                    ys.append(y1)
                    in_zona = False

                else:
                    x1, y1 = x0 - delta * gradf.elems[0], y0 - delta*gradf.elems[1]
                    xs.append(x1)
                    ys.append(y1)
                    in_zona = True
            
            gradf.elems = [self.grad_call(x1, y1)[0],self.grad_call(x1, y1)[1]]
            gradg.elems = [self.g.grad_call(x1, y1)[0],self.g.grad_call(x1, y1)[1]]

            
            #Colinéaire ?
            col = gradf.versor().inner(gradg.versor())
            if not in_zona and abs(col - 1) < tol:
                break
            if in_zona and abs(gradf.elems[0]) < epsilon and abs(gradf.elems[1]) < epsilon:
                break
            
            x0, y0 = x1, y1
            iteration += 1
  
        if returnAsPoints:
            return xs, ys
        else:
            return xs[-1], ys[-1]
        
    def plotTraj(self, x0, y0, ax, fig, xmin, xmax, ymin, ymax, nx, ny, rep = 10000, learning = 0.001):
        
        start_time = time.time()

        # Curvas de nivel de f
        print("Graficando Contours")
        self.contours(xmin = xmin, xmax = xmax, ax=ax, fig=fig)


        # Restriction
        print("Graficando restriccion")
        self.g.contour(0, xmin = xmin, xmax = xmax, ax=ax, fig=fig, color="g-")


        # Coloramos g(x,y)>0
        x = my_linspace(xmin, xmax, 400) 
        y = my_linspace(xmin, xmax, 400) 
        X, Y = np.meshgrid(x, y) 
        Z = self.g.f(X, Y) 
        ax.contourf(X, Y, Z, levels=[0, Z.max()], colors=['green'], alpha=0.5)  # Colorie en rouge la zone où g(x, y) > 0
        
        # Plot los puntos de la busqueda del minimum
        print("Busqueda del minimo con restriccion")
        points = self.get_minimum_2(x0, y0, returnAsPoints=True)


        print("Minimum : ", points[0][-1], points[1][-1])
        self.campo_gradiente(xmin, xmax, ymin, ymax, nx=10, ny=10, ax=ax, fig=fig,points =points)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()   
        print(f'Total time : {(-start_time + time.time()):.2f} seconds')

class opt2d_multiple_constraint(opt2d):
    
    def __init__(self, u, restrictionsEquality, restrictionsInequality):
        super().__init__(u)
        self.restrictionsEquality = [opt2d(g) for g in restrictionsEquality]
        self.restrictionsInequality = [opt2d(g) for g in restrictionsInequality]
        self.allRestric = self.restrictionsInequality + self.restrictionsEquality
        if len(self.restrictionsInequality)==0:
            def zona(x,y):
                salida = 0
                for g in self.allRestric:
                    salida += min(g.f(x,y), 0)**2
                return salida
        else:
            def zona(x,y):
                salida = 0
                for g in self.allRestric:
                    valueG = g.f(x,y)
                    minimum = min(valueG, 0)
                    salida += minimum
                return abs(salida)
        self.consumidor = opt2d_constraint(u, zona)
              
    def get_minimum_1equality_manyInequalities(self, x0, y0, tol=0.0001, epsilon=0.001,delta=0.001, max_iter=100000, returnAsPoints=False):
        
        self.h = self.restrictionsEquality[0]
        
        #Encontrar un punto en la restriccion
        def h_cuadrado(x,y):
            return self.h.f(x,y)**2
        h_cuadrado = opt2d(h_cuadrado)
        xs, ys = h_cuadrado.gdescent(x0, y0, returnAsPoints=True)        
        x0, y0 = xs[-1], ys[-1]
        
        def get_gradients(x, y):
            gradu = vector([self.grad_call(x, y)[0], self.grad_call(x, y)[1]])
            grads = [vector([g.grad_call(x, y)[0], g.grad_call(x, y)[1]]) for g in self.restrictionsInequality]
            gradh = vector([self.h.grad_call(x, y)[0], self.h.grad_call(x, y)[1]])
            return gradu, grads, gradh
        
        iteration = 0
        while iteration < max_iter:
            
            gradu, grads, gradh = get_gradients(x0, y0)
            active_constraints = [g for g in self.restrictionsInequality if abs(g.f(x0, y0)) < epsilon]

            # Si somos en h y no hay ninguna otra restriccion activa
            if len(active_constraints)==0:

                v = (gradu.elems[0] - (gradh.versor().inner(gradu) * gradh.versor()).elems[0],
                     gradu.elems[1] - (gradh.versor().inner(gradu) * gradh.versor()).elems[1])
                
                x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
          
            # Si una otra es activa, es decir gi(x0,y0)=0
            else:
                
                gradg = grads[self.restrictionsInequality.index(active_constraints[0])]

                # Si gi et w apuntan en la misma direccion
                if gradu.inner(gradg)>0:
                    
                    v = (gradu.elems[0] - (gradh.inner(gradu) * gradh.versor()).elems[0],
                         gradu.elems[1] - (gradh.inner(gradu) * gradh.versor()).elems[1])
                   
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                 
                #Si no apuntan en la misma direccion
                else:
                    break
                
            xs.append(x1)
            ys.append(y1)
            
            #Colinéaire ?
            col = gradu.minus().versor().inner(gradh.versor())
            if abs(col - 1) < tol:
                break
            
            x0, y0 = x1, y1
            iteration += 1

        if returnAsPoints:
            return xs, ys
        else:
            return xs[-1], ys[-1]
        
    def get_minimum_only_inequalities(self, x0, y0, tol=0.001, epsilon=0.001, delta=0.01, max_iter=100000, returnAsPoints=False):
        xs, ys = self.find_zona_admissible_2(function=self.consumidor.g, x0=x0, y0=y0, returnAsPoints=True)
        x0, y0 = xs[-1], ys[-1]
        iteration = 0

        def get_gradients(x, y):
            gradu = vector([self.grad_call(x, y)[0], self.grad_call(x, y)[1]])
            grads = [vector([g.grad_call(x, y)[0], g.grad_call(x, y)[1]]) for g in self.restrictionsInequality]
            return gradu, grads
    
        while iteration < max_iter:
            
            gradu, grads = get_gradients(x0, y0)
            active_constraints = [g for g in self.restrictionsInequality if abs(g.f(x0, y0)) < epsilon]
    
            if len(active_constraints) == 0:
                # Zone admissible, suivre le gradient de la fonction objective
                x1, y1 = x0 + delta * gradu.elems[0], y0 + delta * gradu.elems[1]
                in_zona = True
            elif len(active_constraints) == 1:
                in_zona = False
                # Sur une bordure, se déplacer le long de cette restriction
                gradg = grads[self.restrictionsInequality.index(active_constraints[0])]
                if gradu.inner(gradg) < 0:
                    v = (gradu.elems[0] - gradg.versor().inner(gradu) * gradg.versor().elems[0],
                         gradu.elems[1] - gradg.versor().inner(gradu) * gradg.versor().elems[1])
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                else:
                    x1, y1 = x0 + delta * gradu.elems[0], y0 + delta * gradu.elems[1]
            elif len(active_constraints) == 2:
                in_zona = False
                # Intersection de deux restrictions
                gradg1 = grads[self.restrictionsInequality.index(active_constraints[0])]
                gradg2 = grads[self.restrictionsInequality.index(active_constraints[1])]
                if gradu.inner(gradg1) > 0 and gradu.inner(gradg2) > 0:
                    # Si les deux gradients pointent dans la zone admissible
                    x1, y1 = x0 + delta * gradu.elems[0], y0 + delta * gradu.elems[1]
                elif gradu.inner(gradg1) > 0 and gradu.inner(gradg2) < 0:
                    # Si seul gradg1 pointe dans la zone admissible
                    v = (gradu.elems[0] - gradg1.versor().inner(gradu) * gradg1.versor().elems[0],
                         gradu.elems[1] - gradg1.versor().inner(gradu) * gradg1.versor().elems[1])
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                elif gradu.inner(gradg1) < 0 and gradu.inner(gradg2) > 0:
                    # Si seul gradg2 pointe dans la zone admissible
                    v = (gradu.elems[0] - gradg2.versor().inner(gradu) * gradg2.versor().elems[0],
                         gradu.elems[1] - gradg2.versor().inner(gradu) * gradg2.versor().elems[1])
                    x1, y1 = x0 + delta * v[0], y0 + delta * v[1]
                else:
                    # Les deux gradients pointent hors de la zone admissible, s'arrêter
                    break
    
            xs.append(x1)
            ys.append(y1)

            x0, y0 = x1, y1
            iteration += 1
    
        if returnAsPoints:
            return xs, ys
        else:
            return xs[-1], ys[-1]

    def plotTraj(self, x0, y0, ax, fig, xmin, xmax, ymin, ymax, nx, ny, repContour = 10000, learningContour = 0.001,repMin = 10000, learningMin = 0.001, isUtilidad = True, contour_start = 1, contour_stop = 5, contour_step=1):
        
        start_time = time.time()

        # Curvas de nivel de f
        print("Graficando Contours")
        self.contours(start = contour_start, stop = contour_stop, step=contour_step, xmin = xmin, xmax = xmax, ymin = ymin, ymax=ymax, ax=ax, fig=fig, rep=repContour, learning=learningContour, isUtilidad=isUtilidad)


        # Restriction
        print("Graficando restriccion")
        self.consumidor.g.contour(0, xmin = xmin, xmax = xmax, ymin=ymin, ymax=ymax, ax=ax, fig=fig, color="g-", isUtilidad=isUtilidad, rep=repContour, learning=learningContour)


        # Plot los puntos de la busqueda del minimum
        print("Busqueda del minimo con restriccion")
        if len(self.restrictionsEquality) == 1:
            points = self.get_minimum_1equality_manyInequalities(x0, y0, returnAsPoints=True)
        else:
            points = self.get_minimum_only_inequalities(x0, y0, returnAsPoints=True, delta=learningMin, max_iter=repMin, epsilon=0.01)
        print("Minimum : ", points[0][-1], points[1][-1])
        
        self.campo_gradiente(xmin, xmax, ymin, ymax, nx=10, ny=10, ax=ax, fig=fig,points =points)
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()   
        print(f'Total time : {(-start_time + time.time()):.2f} seconds')
    
    def find_zona_admissible_2(self, function, x0=1,y0=4,delta=0.001,tol=0.001,Nmax=100000, returnAsPoints = False):
         # Récupération des fonctions de dérivées partielles
         
         gradf = function.grad_call(x0, y0)
         learningrate = delta
         xs = [x0]
         ys = [y0]
         
         count = 0
         in_zona = False
         
         # Vérifier les conditions d'arrêt 
         if function.f(x0, y0) < tol:
             in_zona = True
             
         while count < Nmax and not in_zona:
             # Calcul des nouvelles valeurs des variables
             
             x1 = x0 - learningrate * gradf[0]
             y1 = y0 - learningrate * gradf[1]
            
             
             # Ajouter les nouvelles valeurs aux listes
             xs.append(x1)
             ys.append(y1)
             
            
             # Vérifier les conditions d'arrêt 
             if function.f(x1, y1) < tol:
                 break
             

             # Mettre à jour les variables pour la prochaine itération
             x0, y0 = x1, y1
             gradf = function.grad_call(x0, y0)
             count += 1
         
         if not returnAsPoints:        
             return (xs[-1], ys[-1])

         else:
             return xs, ys
              
class polytope():
    def __init__(self,P_list,f) :
        self.P= P_list
        self.P.sort()
        self.f= f
        if not self.check_point():raise ValueError("Problema con los puntos")
        
    def puntos(self):
        x1 = self.f(self.P[0][0], self.P[0][1])
        x2 = self.f(self.P[1][0], self.P[1][1])
        x3 = self.f(self.P[2][0], self.P[2][1])
        
        lista = [x1, x2, x3]
        lista_puntos = [0, 0, 0]
        pos = [0, 1, 2]
        for i in range(len(lista)):
            if max(lista) == lista[i]:
                lista_puntos[0] = self.P[i]
                max_p = i
            if min(lista) == lista[i]:
                lista_puntos[2] = self.P[i]
                min_p = i
        
        pos_maxmin = [max_p, min_p]
        mid_pos = list(set(pos) - set(pos_maxmin))[0]
        lista_puntos[1] = self.P[mid_pos]
        
        self.P = lista_puntos
        
    def check_point(self):
        self.puntos()
        vec_1 = (self.P[0][0] - self.P[1][0], self.P[0][1] - self.P[1][1])
        vec_2 = (self.P[2][0] - self.P[1][0], self.P[2][1] - self.P[1][1])
        
        lista_matriz = [vec_1[0], vec_1[1], vec_2[0], vec_2[1]]
        matriz = myarray(lista_matriz, 2, 2, True)
        
        if matriz.det() == 0:
            print("NOT OK : Los puntos proporcionados no forman un triángulo")
            return False
        else:
            return True
        
    def recta(self, P, Q):
        lista_matriz = [P[0], 1, Q[0], 1]
        lista_sol = [P[1], Q[1]]
        sistema = lineq(lista_matriz, 2, True, lista_sol)
        
        sol_sist = sistema.solve().elems
        
        return sol_sist
    
    def perpendicular(self, P, Q, P_ref):
        recta = self.recta(P, Q)
        b = P_ref[1] + (1 / recta[0]) * P_ref[0]
        
        perpendicular = [-1 / recta[0], b]
        
        return perpendicular
    
    def interseccion(self, P, Q, P_ref):
        recta = self.recta(P, Q)
        perpendicular = self.perpendicular(P, Q, P_ref)
        
        x = (perpendicular[1] - recta[1]) / (recta[0] - perpendicular[0])
        y = recta[0] * x + recta[1]
        
        salida = (x, y)
        
        return salida
    
    def polyt_prog(self, epsilon=0.0000001):
        self.puntos()
        inter = self.interseccion(self.P[1], self.P[2], self.P[0])
        
        x_ref = inter[0] - (self.P[0][0] - inter[0])
        y_ref = inter[1] - (self.P[0][1] - inter[1])
        
        x_prime = (x_ref, y_ref)
        
        restric = (self.P[0][0] - self.P[2][0], self.P[0][1] - self.P[2][1])
        
        while (abs(restric[0]) > epsilon and abs(restric[1]) > epsilon) or (abs(self.f(self.P[0][0], self.P[0][1]) - self.f(self.P[2][0], self.P[2][1])) > epsilon):
            if self.f(x_prime[0], x_prime[1]) < self.f(self.P[0][0], self.P[0][1]):
                self.P[0] = x_prime
                self.P.sort()
                self.puntos()
                inter = self.interseccion(self.P[1], self.P[2], self.P[0])
                
                x_ref = inter[0] - (self.P[0][0] - inter[0])
                y_ref = inter[1] - (self.P[0][1] - inter[1])
                
                x_prime = (x_ref, y_ref)

            else:
                inter = self.interseccion(self.P[0], self.P[2], self.P[1])
                
                x_ref = inter[0] - (self.P[1][0] - inter[0])
                y_ref = inter[1] - (self.P[1][1] - inter[1])
                
                x_prime = (x_ref, y_ref)
                
                if self.f(x_prime[0], x_prime[1]) < self.f(self.P[1][0], self.P[1][1]):
                    self.P[1] = x_prime
                    self.P.sort()
                    self.puntos()
                    inter = self.interseccion(self.P[1], self.P[2], self.P[0])
                    x_ref = inter[0] - (self.P[0][0] - inter[0])
                    y_ref = inter[1] - (self.P[0][1] - inter[1])
                    
                    x_prime = (x_ref, y_ref)

                else:
                    self.P[0] = ((1/2) * (self.P[0][0] + self.P[2][0]), (1/2) * (self.P[0][1] + self.P[2][1]))
                    self.P[1] = ((1/2) * (self.P[1][0] + self.P[2][0]), (1/2) * (self.P[1][1] + self.P[2][1]))
                    
                    inter = self.interseccion(self.P[1], self.P[2], self.P[0])
                    
                    x_ref = inter[0] - (self.P[0][0] - inter[0])
                    y_ref = inter[1] - (self.P[0][1] - inter[1])
                    
                    x_prime = (x_ref, y_ref)

            
            restric = (self.P[0][0] - self.P[2][0], self.P[0][1] - self.P[2][1])
        
        print(self.P[0])
        return self.P[0]
    

# Definimos la función objetivo
def objective(x):
    return np.sqrt((x[0] - x[2])**2 + (x[1] - x[3])**2)

# Definimos las restricciones
def constraint_circle(x):
    return 1 - ((x[0] - 3)**2 + x[1]**2)

def constraint_square(x):
    return [
        1 - x[2] - x[3],
        x[3] - x[2] + 1,
        1 - x[3] + x[2],
        x[2] + x[3] + 1
    ]

# Gradiente de la función objetivo
def grad_objective(x):
    grad = np.zeros_like(x)
    dist = objective(x)
    grad[0] = (x[0] - x[2]) / dist
    grad[1] = (x[1] - x[3]) / dist
    grad[2] = (x[2] - x[0]) / dist
    grad[3] = (x[3] - x[1]) / dist
    return grad

# Proyectar un punto sobre el círculo
def project_circle(x):
    angle = np.arctan2(x[1], x[0] - 3)
    return np.array([3 + np.cos(angle), np.sin(angle)])

# Proyectar un punto sobre el cuadrado
def project_square(x):
    if x[0] + x[1] > 1:
        x[0] = 1 - x[1]
    if x[1] - x[0] > 1:
        x[1] = 1 + x[0]
    if x[1] + x[0] < -1:
        x[0] = -1 - x[1]
    if x[0] + x[1] < -1:
        x[1] = -1 - x[0]
    return x

# Método de descenso por gradientes
def gradient_descent(x0, lr=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)  # Aseguramos que x0 sea de tipo float
    for _ in range(max_iter):
        grad = grad_objective(x)
        x -= lr * grad
        x[:2] = project_circle(x[:2])
        x[2:] = project_square(x[2:])
        if np.linalg.norm(grad) < tol:
            break
    return x

# Puntos iniciales para (x1, x2, x3, x4)
x0 = [3, 0, 1, 0]

# Ejecutamos el método de descenso por gradientes
result_x = gradient_descent(x0)
result_fun = objective(result_x)
result_success = constraint_circle(result_x) >= 0 and all(c >= 0 for c in constraint_square(result_x))

result_x, result_fun, result_success