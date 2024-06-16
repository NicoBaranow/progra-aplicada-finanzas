import numpy as np
import matplotlib.pyplot as plt

plt.arrow(0, 0, 1, 1, head_width=0.05, head_length=0.1, fc='r', ec='r', color='b')


plt.show()

class opt2d:
    def __init__(self, f):

        self.f = f
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
        Devuelve el gradiente de f en forma de tupla
        '''
        return (self.fx(x), self.fy(x))

    def fv(self, x = (0, 0), v = (2, 3)):
        '''
        Dado un punto (x,y) y un vector v, devuelve la derivada direccional de f en esa dirección
        '''

        grad = self.gradf(x)
        norma = (v[0] ** 2 + v[1] ** 2) ** 0.5
        return grad[0]*v[0]/norma + grad[1]*v[1]/norma

    def campo_gradiente(self, x_range = (0,10), y_range = (0,10), nx = (0,0), ny = (0,0)):
        '''
        Toma un rango en x, un rango en y y la cantidad de puntos en x e y
        '''
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        U = np.zeros(X.shape)
        V = np.zeros(Y.shape)

        for i in range(nx):
            for j in range(ny):
                grad = self.gradf((X[i, j], Y[i, j]))
                U[i, j] = grad[0]
                V[i, j] = grad[1]

        plt.quiver(X, Y, U, V)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Campo de Gradientes')
        plt.show()

    def gdescent(self, x0, y0, delta=0.01, tol=0.001, Nmax=100000):
        x = np.array([x0, y0])
        path = [x]
        for _ in range(Nmax):
            grad = self.gradf(x)
            x_new = x - delta * grad
            path.append(x_new)
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x, np.array(path)

# Función de ejemplo
def function(x): #x es una tupla (x,y)
    return np.sin(x[0]/3) * np.cos(x[1]/3)

# Crear una instancia de la clase
opt = opt2d(function) #le pasamos una función como parametro, pero no la ejecutamos, por eso sin ()

print(opt.fv())

