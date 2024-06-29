import matplotlib.pyplot as plt
import numpy as np
from matrices_suggested import lineq, myarray
from vectores import vector

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
            print("Los puntos proporcionados no forman un triángulo")
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

