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

    def rootfind(self, func, x0, tolerance=1e-6, max_iterations=1000):
        x = x0
        for _ in range(max_iterations):
            fx = func(x)
            f_prime_x = (func(x + self.hx) - func(x - self.hx)) / (2 * self.hx) #Correccion del calc de la derivada
            if f_prime_x == 0:
                raise ValueError("Derivative is zero, no convergence possible.")
            x_new = x - fx / f_prime_x
            if abs(x_new - x) < tolerance:
                return x_new
            x = x_new
        raise ValueError("Did not converge.")

    def contour1(self, x0, y0, xmin, xmax):
        # Definir el valor de k
        k = self.f(x0, y0)
        
        # Crear una lista de puntos
        xs = my_linspace(xmin, xmax, 50)
        ys = []
        
        # g(x,y) = f(x, y) - k
        def g(y, x):
            return self.f(x, y) - k
        
        # Método de Newton-Raphson
        for x in xs:
            try:
                if len(ys) == 0:

                    y_initial = y0
                    last_good = y0
                else:

                    if not np.isnan(ys[-1]):
                        y_initial = ys[-1]
                        last_good = ys[-1]
                    else:
                        y_initial = last_good
                
                y = newton_raphson(lambda y: g(y, x), y_initial)
                
                ys.append(y)
            except ValueError as e:
                ys.append(np.nan)
        
        # Filtramos Nan values
        xs_filtered = [x for x, y in zip(xs, ys) if not np.isnan(y)]
        ys_filtered = [y for y in ys if not np.isnan(y)]
        

        if ys_filtered: 
            fig, ax = plt.subplots()
            ax.plot(xs_filtered, ys_filtered, 'ro-', markersize=1, linewidth=1)
            
            ax.set_xlim([-7, 7])
            ax.set_ylim([-7, 7])
            ax.set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.show()
        else:
            print("Aucune valeur valide trouvée pour tracer la courbe de niveau.")

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
            
            # #Colinéaire ?
            # colinearios = [gradu.minus().versor().inner(gradg.versor()) for gradg in grads]
            # stop = False
            # for col in colinearios:
            #     if not in_zona and (abs(col - 1) < tol):
            #         stop = True
            #     if in_zona and abs(gradu.elems[0]) < epsilon and abs(gradu.elems[1]) < epsilon:
            #         stop = True
            
            # if stop: 
            #     break
    
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
    
    
def tp1_1(x,y):
    return np.sin((9 * x**2 + 6 * x * y - 33 * x - 26 * y + 4 * y**2 + 133) / 100)

def tp1_2(x,y):
    return (myarray([x,y], 1, 2) * myarray([9,3,3,4], 2, 2) * myarray([x,y], 2, 1)).elems[0]

def test(x,y):
    return (x-2)**2 + (y-5)**2 + 3

def newton_raphson(func, x0, hx=1e-6, tol=1e-4, max_iter=10000):
   
    x = x0
    for i in range(max_iter):
        fx = func(x)
        f_prime_x = (func(x + hx) - func(x - hx)) / (2 * hx)  # Approximation numérique de la dérivée
        
        if f_prime_x == 0:
            raise ValueError("La dérivée est nulle, pas de convergence possible.")
        
        x_new = x - fx / f_prime_x
        
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    
    raise ValueError("La méthode de Newton-Raphson n'a pas convergé.")



# Trois façons de trouver un minimum
   # 1 : tu traces le champ gradient et tu regardes les flèches
   # 2 : tu appelles polytope
   # 3 : tu appelles gdescent

#%% TP Optim 1

# Nouvelle instance d'optimizer avec la fonction tp(x,y)
def TP_Optim1():
    optimizer = opt2d(tp1_2)
    optimizer.contours()

    # On trouve le minimum avec la méthode polytope
    pltp = polytope([(0.5,0.21),(0.01,0.01),(0.8,0.5)],test)
    print(" \nAvec Polytope")
    pt = pltp.polyt_prog()
    print("Gradient de f : ", optimizer.grad_call(pt[0], pt[1]))
    
    # Avec le champ gradient
    optimizer.campo_gradiente(-5, 5, -5, 5, 20, 20, optimizer.gdescent(1, 1, returnAsPoints=True))
    
    
    # Avec gdescent
    print("\nAvec gdescent Global")
    pt = optimizer.gdescent_global(x0=-2, y0=-2)
    print("Gradient de f : ", optimizer.grad_call(pt[0], pt[1]))
    print(optimizer.minimumState(optimizer.hessienna(pt[0], pt[1])))


#%% TP Optim 2

# Optimisation sans contrainte

def TP_2_SC_1(x, y):
    return 2 * (x**3) + (x * (y**2)) + (5 * (x**2)) + y**2

def TP_2_SC_2(x, y):
    return x / (1 + x**2 + y**2)

def TP_2_SC_3(x, y):
    return x * np.sin(y)

def TP_2_SC_4(x, y):
    return np.cos((x**2 + y**2) / 10) * np.exp(-x**2)

# On appelle toutes les fonctions
def TP_Optim_SC():
    opt = opt2d(TP_2_SC_1)
    print(opt.gdescent(x0=2, y0 = 3))
    
    opt = opt2d(TP_2_SC_2)
    print(opt.gdescent(2,3))
    
    # opt = opt2d(TP_2_SC_3)
    # opt.contours()
    # opt.gdescent_global()
    
    # opt = opt2d(TP_2_SC_4)
    # opt.contours()
    # opt.gdescent_global()

# Optimisation avec contrainte


def gg(x,y):
    return y**3 - x**2

def ff(x,y):
    return y

def TP_2_AC_1(x, y):
    return x*y

def TP_2_AC_2(x, y):
    return (1/x) + 1/y

def TP_2_AC_3(x, y):
    return x + y

def TP_2_AC_4(x, y):
    return x**2 + y**2

def restric1(x,y):
    return -(x**2 + y**2 - 4)

def restric2(x,y):
    return (1/x)**2 + (1/y)**2 - (1/4)**2

def restric3(x,y):
    return 16

def restric4(x,y):
    return (x-1)**3 - y**2

def TP_Optim_AC():

    opt = opt2d_constraint(TP_2_AC_1, restric1)

    # Campo gradiente de f + curva nivel de f + restriccion g
    opt.fig1, opt.ax1 = plt.subplots()
    opt.plot1(opt.ax1, opt.fig1,-5,5,-5,5,10,10)
    
    # Campo gradiente de g + curva nivel de g 
    opt.fig2, opt.ax2 = plt.subplots()
    opt.plot2(opt.ax2, opt.fig2,-5,5,-5,5,10,10,2,2)
    
    # Curvas nivel de f + restriction g
    opt.fig3, opt.ax3 = plt.subplots()
    opt.plot3(opt.ax3, opt.fig3,-5,5,-5,5,10,10,2,2)
    
    #Trouver minimum sous contrainte et afficher trajectoire
    opt.fig4, opt.ax4 = plt.subplots()
    opt.plot4(3,7, opt.ax4, opt.fig4,-5,5,-5,5,10,10)
    

    
    # opt = opt2d_constraint(TP_2_AC_2, restric2)

    # # Campo gradiente de f + curva nivel de f + restriccion g
    # opt.fig1, opt.ax1 = plt.subplots()
    # opt.plot1(opt.ax1, opt.fig1,-5,5,-5,5,10,10)
    
    # # Campo gradiente de g + curva nivel de g 
    # opt.fig2, opt.ax2 = plt.subplots()
    # #opt.plot2(opt.ax2, opt.fig2,-5,5,-5,5,10,10,2,2)
    
    # # Curvas nivel de f + restriction g
    # opt.fig3, opt.ax3 = plt.subplots()
    # #opt.plot3(opt.ax3, opt.fig3,-5,5,-5,5,10,10,2,2)
    
    # #Trouver minimum sous contrainte et afficher trajectoire
    # opt.fig4, opt.ax4 = plt.subplots()
    # opt.plot4(2,-2, opt.ax4, opt.fig4,-5,5,-5,5,10,10)
    
    
    # opt = opt2d_constraint(TP_2_AC_3, restric3)

    # # Campo gradiente de f + curva nivel de f + restriccion g
    # opt.fig1, opt.ax1 = plt.subplots()
    # opt.plot1(opt.ax1, opt.fig1,-5,5,-5,5,10,10)
    
    # # Campo gradiente de g + curva nivel de g 
    # opt.fig2, opt.ax2 = plt.subplots()
    # #opt.plot2(opt.ax2, opt.fig2,-5,5,-5,5,10,10,2,2)
    
    # # Curvas nivel de f + restriction g
    # opt.fig3, opt.ax3 = plt.subplots()
    # #opt.plot3(opt.ax3, opt.fig3,-5,5,-5,5,10,10,2,2)
    
    # #Trouver minimum sous contrainte et afficher trajectoire
    # opt.fig4, opt.ax4 = plt.subplots()
    # opt.plot4(2,3, opt.ax4, opt.fig4,-5,5,-5,5,10,10)
    
    
    
    opt = opt2d_constraint(TP_2_AC_4, restric4)

    # Campo gradiente de f + curva nivel de f + restriccion g
    # opt.fig1, opt.ax1 = plt.subplots()
    # opt.plot1(opt.ax1, opt.fig1,-5,5,-5,5,10,10)
    
    # Campo gradiente de g + curva nivel de g 
    # opt.fig2, opt.ax2 = plt.subplots()
    #opt.plot2(opt.ax2, opt.fig2,-5,5,-5,5,10,10,2,2)
    
    # Curvas nivel de f + restriction g
    # opt.fig3, opt.ax3 = plt.subplots()
    #opt.plot3(opt.ax3, opt.fig3,-5,5,-5,5,10,10,2,2)
    
    #Trouver minimum sous contrainte et afficher trajectoire
    opt.fig4, opt.ax4 = plt.subplots()
    opt.plot4(4,-2.2, opt.ax4, opt.fig4,-5,5,-5,5,10,10)
    

    
#%% 

def TP_Optim_AC_inequality():
    opt = opt2d_constraint_inequality(TP_2_AC_1, restric1)
    fig, ax = plt.subplots()
    opt.plotTraj(-1.1,-1,ax,fig,-5, 5, -5, 5, 10, 10)
    fig, ax = plt.subplots()
    opt.plotTraj(3.8,1,ax,fig,-5, 5, -5, 5, 10, 10)
    fig, ax = plt.subplots()
    opt.plotTraj(3,3,ax,fig,-5, 5, -5, 5, 10, 10)
    fig, ax = plt.subplots()
    opt.plotTraj(-3,0.5,ax,fig,-5, 5, -5, 5, 10, 10)
    


# Fonction principale pour l'optimisation
def TP_multivariada_a():
    
    def g1(x, y):
        return x

    def g2(x, y):
        return y

    def g3(x, y, px, py, m):
        return m - (px * x + py * y)

    def u(x, y):
        if x <= 0 or y <= 0:
            return float('nan')
        return x**0.5 * y**0.5
    
    opt = opt2d_multiple_constraint(u, [lambda x,y:g3(x,y,1,1,10)], [g1, g2])
    fig, ax = plt.subplots()
    opt.plotTraj(x0=1, y0=6, ax=ax, fig=fig, xmin=0, xmax=10, ymin=0, ymax=10, nx=10, ny=10, learningMin =0.001,repMin=100000)
    u_star = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
    
    # Maintenir le revenu constant et faire varier py
    py_values = [0.7 + i * 0.05 for i in range(13)]
    x_star_py = []
    y_star_py = []
    for py in py_values:
        px = 1
        m = 10
        g3_current = lambda x, y: g3(x, y, px, py, m)
        opt = opt2d_multiple_constraint(u, [], [g1, g2, g3_current])
        maxim = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
        x_star_py.append(maxim[0])
        y_star_py.append(maxim[1])
        print(f'Minimo para px = 1 y py = {py} : {maxim[0], maxim[1]}')
        

    plt.figure()
    plt.plot(py_values, x_star_py, label='x*')
    plt.plot(py_values, y_star_py, label='y*')
    plt.xlabel('py')
    plt.ylabel('Valores óptimos')
    plt.legend()
    plt.title('Valores óptimos en función de py')
    plt.show()
    
    # Maintenir les prix constants et faire varier beta
    beta_values = [0.2 + i * 0.05 for i in range(13)]
    x_star_beta = []
    y_star_beta = []
    for beta in beta_values:
        u_current = lambda x, y: x**0.5 * y**beta
        opt = opt2d_multiple_constraint(u_current, [], [g1, g2, lambda x, y: g3(x, y, 1, 1, 10)])
        maxim = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
        x_star_beta.append(maxim[0])
        y_star_beta.append(maxim[1])

    plt.figure()
    plt.plot(beta_values, x_star_beta, label='x*')
    plt.plot(beta_values, y_star_beta, label='y*')
    plt.xlabel('beta')
    plt.ylabel('Valores óptimos')
    plt.legend()
    plt.title('Valores óptimos en función de beta')
    plt.show()
    
    
def TP_multivariada_b():
   
    # Minimisar canasta
    def g1(x, y):
        return x

    def g2(x, y):
        return y

    def g3(x, y, px, py, m):
        return m - (px * x + py * y)

    def u(x, y):
        return min((x**0.5),(y**0.5))
    
    opt = opt2d_multiple_constraint(u, [lambda x,y:g3(x,y,1,1,10)], [g1, g2])
    fig, ax = plt.subplots()
    opt.plotTraj(x0=1, y0=6, ax=ax, fig=fig, xmin=0.5, xmax=10, ymin=0.5, ymax=10, nx=10, ny=10, learningMin =0.001,repMin=100000)
 
    # Maintenir le revenu constant et faire varier py
    py_values = [0.7 + i * 0.05 for i in range(13)]
    x_star_py = []
    y_star_py = []
    for py in py_values:
        px = 1
        m = 10
        g3_current = lambda x, y: g3(x, y, px, py, m)
        opt = opt2d_multiple_constraint(u, [], [g1, g2, g3_current])
        maxim = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
        x_star_py.append(maxim[0])
        y_star_py.append(maxim[1])
        print(f'Minimo para px = 1 y py = {py} : {maxim[0], maxim[1]}')
        

    plt.figure()
    plt.plot(py_values, x_star_py, label='x*')
    plt.plot(py_values, y_star_py, label='y*')
    plt.xlabel('py')
    plt.ylabel('Valores óptimos')
    plt.legend()
    plt.title('Valores óptimos en función de py')
    plt.show()
    
    # Maintenir les prix constants et faire varier beta
    beta_values = [0.2 + i * 0.05 for i in range(13)]
    x_star_beta = []
    y_star_beta = []
    for beta in beta_values:
        u_current = lambda x, y: min(x**0.5, y**beta)
        opt = opt2d_multiple_constraint(u_current, [], [g1, g2, lambda x, y: g3(x, y, 1, 1, 10)])
        maxim = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
        x_star_beta.append(maxim[0])
        y_star_beta.append(maxim[1])

    plt.figure()
    plt.plot(beta_values, x_star_beta, label='x*')
    plt.plot(beta_values, y_star_beta, label='y*')
    plt.xlabel('beta')
    plt.ylabel('Valores óptimos')
    plt.legend()
    plt.title('Valores óptimos en función de beta')
    plt.show()
    
    
def TP_multivariada_d():
    
    # Recuperar u_star
    def g1(x, y):
        return x

    def g2(x, y):
        return y

    def g3(x, y, px, py, m):
        return m - (px * x + py * y)

    def u(x, y):
        if x <= 0 or y <= 0:
            return float('nan')
        return x**0.5 * y**0.5
    
    opt = opt2d_multiple_constraint(u, [], [g1, g2, lambda x,y:g3(x,y,1,1,10)])
    minimo = opt.get_minimum_only_inequalities(5, 5, returnAsPoints=False, delta=0.001,max_iter=10000, epsilon=0.01)
    u_star = opt.f(minimo[0], minimo[1])
    
    
    def g1(x, y):
        return x

    def g2(x, y):
        return y

    def g3(x, y):
        return (x**0.5 * y**0.5) - u_star

    def u(x, y):
        return -(x + y)
    
    opt = opt2d_multiple_constraint(u, [], [g1, g2, g3])
    fig, ax = plt.subplots()
    opt.plotTraj(x0=1, y0=6, ax=ax, fig=fig, xmin=0.5, xmax=20, ymin=0.5, ymax=20, nx=10, ny=10, learningMin =0.001,repMin=100000)
 
   
def TP_multivariada_f():
    def utilidad(x,y):
        return (x+y)
    def g1(x,y):
        return 5 - (0.2*x + 0.1*y)
    def g2(x,y):
        return 1 - (0.025*x + 0.050*y)
    def g3(x,y):
        return x
    def g4(x,y):
        return y
    opt = opt2d_multiple_constraint(utilidad, [], [g1, g2, g3, g4])
    fig, ax = plt.subplots()
    opt.plotTraj(10, 0, ax, fig, 0, 50, 0, 50, 10, 10, learningMin=0.01, repMin=100000, learningContour=0.01, repContour=100000, contour_start=5, contour_stop=21, contour_step=5)

def twocircles():

    def g1(x,y):
        return -((x-2)**2 + (y-2)**2 - 3)
    
    def g2(x,y):
        return -((x-4)**2 + (y-2)**2 - 3)
    
    def f(x,y):
        return x+y
    
    opt = opt2d_multiple_constraint(f, [], [g1, g2])
    figMul, axMul = plt.subplots()
    opt.plotTraj(6, 2, axMul, figMul, xmin=0, xmax=10, ymin=0, ymax=10, nx=10, ny=10,learningMin=0.001, repMin=10000, learningContour=0.001, repContour=10000, isUtilidad=True, contour_start=4, contour_stop=8)


TP_Optim_AC()