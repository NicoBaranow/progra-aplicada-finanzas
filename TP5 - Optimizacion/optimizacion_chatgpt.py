import numpy as np
import matplotlib.pyplot as plt

class opt2d:
    def __init__(self, f):

        self.f = f
        self.hx = 0.001
        self.hy = 0.001

    def fx(self, x):
        return (self.f(x[0] + self.hx, x[1]) - self.f(x[0] - self.hx, x[1])) / (2 * self.hx)

    def fy(self, x):
        return (self.f(x[0], x[1] + self.hy) - self.f(x[0], x[1] - self.hy)) / (2 * self.hy)

    def fxx(self, x):
        return (self.fx((x[0] + self.hx, x[1])) - self.fx((x[0] - self.hx, x[1]))) / (2 * self.hx)

    def fyy(self, x):
        return (self.fy((x[0], x[1] + self.hy)) - self.fy((x[0], x[1] - self.hy))) / (2 * self.hy)

    def fxy(self, x):
        return (self.fx((x[0], x[1] + self.hy)) - self.fx((x[0], x[1] - self.hy))) / (2 * self.hy)

    def gradf(self, x):
        return np.array([self.fx(x), self.fy(x)])

    def fv(self, x, v):
        grad = self.gradf(x)
        return np.dot(grad, v) / np.linalg.norm(v)

    def campo_gradiente(self, x_range, y_range, nx, ny):
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
def f(x, y):
    return np.sin(x/3) * np.cos(y/3)

# Crear una instancia de la clase
opt = opt2d(f)

# Graficar el campo de gradientes
opt.campo_gradiente((-5, 5), (-5, 5), 20, 20)

# Buscar el mínimo local usando descenso por gradiente
minimo, path = opt.gdescent(4, 2)
print("Mínimo local:", minimo)

# Graficar la trayectoria del descenso por gradiente
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'o-', label='Trayectoria')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Descenso por Gradiente')
plt.legend()
plt.show()
