import matplotlib.pyplot as plt
from optimParcial import opt4d

# Parámetros de las posiciones
posiciones = [(10, 0), (0, 1), (0, -2)]
probabilidades = [0.5, 0.2, 0.3]

# Definición de la función objetivo
def distancia_esperada(p):
    x1, y1, x2, y2 = p
    distancias = []
    for (a, b), prob in zip(posiciones, probabilidades):
        d1 = ((a - x1)**2 + (b - y1)**2) ** 0.5
        d2 = ((a - x2)**2 + (b - y2)**2) ** 0.5
        dist_min = min(d1, d2)
        distancias.append(prob * dist_min)
    return sum(distancias)

# Ejecución de la optimización
def optimizar():
    f_objetivo = lambda p: distancia_esperada(p)
    opt = opt4d(f_objetivo)
    x0 = (5, 0, 5, 2)  # Punto inicial en R4
    p_opt, path = opt.gdescent_con_restriccion(x0)
    best_dist = f_objetivo(p_opt)
    return p_opt, best_dist, path

# Ejecutar la optimización
best_point, min_dist, path = optimizar()
x1_opt, y1_opt, x2_opt, y2_opt = best_point
print("La mínima distancia esperada es:", min_dist)
print("Las posiciones óptimas de los pelotones son:", (x1_opt, y1_opt), (x2_opt, y2_opt))
distancia_entre_pelotones = ((x1_opt - x2_opt)**2 + (y1_opt - y2_opt)**2) ** 0.5
print("Distancia entre los pelotones:", distancia_entre_pelotones)

fig, ax = plt.subplots()

# Graficar las posiciones A, B, C
ax.plot(10, 0, 'bo', label='Posición A (10, 0)')
ax.plot(0, 1, 'go', label='Posición B (0, 1)')
ax.plot(0, -2, 'ro', label='Posición C (0, -2)')

# Graficar el descenso por gradiente
if path:
    path_x1, path_y1, path_x2, path_y2 = zip(*path)
    ax.plot(path_x1, path_y1, 'r-', label='Pelotón 1')
    ax.plot(path_x2, path_y2, 'b-', label='Pelotón 2')

# Graficar el punto final
ax.plot(best_point[0], best_point[1], 'rx', label='Posición Final Pelotón 1')
ax.plot(best_point[2], best_point[3], 'bx', label='Posición Final Pelotón 2')

# Configuraciones del gráfico
plt.xlim(-1, 11)
plt.ylim(-3, 3)
plt.title('Optimización de las Posiciones de los Pelotones')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
