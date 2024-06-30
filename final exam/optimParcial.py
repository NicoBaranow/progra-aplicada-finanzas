import matplotlib.pyplot as plt
import numpy as np
from matrices_suggested import myarray

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
        return (self.f(x + self.hx, y) - self.f(x - self.hx, y)) / (2 * self.hx)

    def fy(self, x, y):
        return (self.f(x, y + self.hy) - self.f(x, y - self.hy)) / (2 * self.hy)

    def fxx(self, x, y):
        return (self.f(x + self.hx, y) - 2*self.f(x, y) + self.f(x - self.hx, y)) / (self.hx**2)

    def fxy(self, x, y):
        return (self.f(x + self.hx, y + self.hy) - self.f(x - self.hx, y + self.hy) - self.f(x + self.hx, y - self.hy) + self.f(x - self.hx, y - self.hy)) / (4 * self.hx * self.hy)

    def fyy(self, x, y):
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
                norm = (dx**2 + dy**2) ** 0.5
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
                x0, y0 = puntos_ah[0] 
                for _ in range(rep):
                    grad = self.grad_call(x0, y0)
                    v = (grad[1], - grad[0])
                    x1 = (x0 + learning*v[0], y0 + learning*v[1])
                    x0, y0 = x1
                    puntos_h.append(x1)
        
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
            
            norm = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
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
                
                norm = ((x1 - x0)**2 + (y1 - y0)**2) 
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

class opt4d(opt2d):
    def __init__(self, func, hx=0.001, hy=0.001):
        super().__init__(func, hx, hy)
        self.nd = 4  # Definimos que estamos trabajando en R4

    def grad_call(self, p, h=0.001):
        grad = np.zeros(self.nd)
        for i in range(self.nd):
            p1 = np.array(p, dtype=float)
            p2 = np.array(p, dtype=float)
            p1[i] += h
            p2[i] -= h
            grad[i] = (self.f(p1) - self.f(p2)) / (2 * h)
        return grad

    def gdescent_con_restriccion(self, x0, delta=0.01, tol=0.0001, max_iter=1000):
        historial = []
        p = np.array(x0, dtype=float)
        for i in range(max_iter):
            if self.restriccion(p) >= 0:
                grad = self.grad_call(p)
                p -= delta * grad
                historial.append(p.copy())
                if np.linalg.norm(grad) < tol:
                    break
            else:
                break
        return p, historial

    def restriccion(self, p):
        x1, y1, x2, y2 = p
        return 9 - ((x1 - x2)**2 + (y1 - y2)**2)

class CircleOptimization(opt2d):
    def __init__(self, f, r):
        super().__init__(f)
        self.r = r

    def gdescent_con_restriccion(self, x0, y0, delta=0.01, tol=0.0001, max_iter=1000):
        historial = []
        x, y = x0, y0
        for i in range(max_iter):
            if self.r([x, y]) >= 0:
                fx, fy = self.grad_call(x, y)
                x -= delta * fx
                y -= delta * fy
                historial.append((x, y))
                if (fx**2 + fy**2)**0.5 < tol:
                    break
            else:
                break
        return x, y, historial

class funcion(object):
    def __init__(self,f,nd,name="None"):   
        # f es una función definida fuera de la clase
        # nd cantidad de variables 
        # name tiene la definicion algebraica de la funcion
        
        self.f = f
        self.nd = nd
        self.name = name
    
    def __call__(self,x): #x es una tupla/lista. Sale un escalar
        if len(x)==self.nd:
            salida = self.f(x)
        else:
            print("la funcion " + self.name+ " requiere inputs de dimension "+ str(self.nd))
            salida  = None
        return salida
    
    def grad(self,x,h=0.001): # sale una lista 
        x = list(x)
        salida = [0]*self.nd
        for i in range(self.nd):
            aux = [0]*self.nd 
            aux[i] = h
            salida[i]=(self.f([(x[i]+aux[i]) for i in range(self.nd)])-self.f([(x[i]-aux[i]) for i in range(self.nd)]))/(2*h)
        return salida
