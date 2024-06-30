from optimParcial import funcion, my_linspace, CircleOptimization

centro_circulo = (3, 0)  # Centro del círculo
radio_circulo = 1        # Radio del círculo
vertices_cuadrado = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Vértices del cuadrado

def distancia(p):
    x1, x2, x3, x4 = p
    return ((x1 - x3)**2 + (x2 - x4)**2) ** 0.5

def restriccion_circulo(p):
    x1, x2 = p
    return radio_circulo**2 - ((x1 - centro_circulo[0])**2 + (x2 - centro_circulo[1])**2)

def restriccion_cuadrado_1(p):
    x3, x4 = p
    return 1 - x3 - x4

def restriccion_cuadrado_2(p):
    x3, x4 = p
    return x4 - x3 + 1

def restriccion_cuadrado_3(p):
    x3, x4 = p
    return x3 + x4 + 1

def restriccion_cuadrado_4(p):
    x3, x4 = p
    return x3 - x4 + 1

# Crear instancias de la clase `funcion` para cada restricción
f_distancia = funcion(lambda p: distancia(p), 4, "Distancia")
r_circulo = funcion(lambda p: restriccion_circulo(p), 2, "Restriccion_Circulo")
r_cuadrado_1 = funcion(lambda p: restriccion_cuadrado_1(p), 2, "Restriccion_Cuadrado_1")
r_cuadrado_2 = funcion(lambda p: restriccion_cuadrado_2(p), 2, "Restriccion_Cuadrado_2")
r_cuadrado_3 = funcion(lambda p: restriccion_cuadrado_3(p), 2, "Restriccion_Cuadrado_3")
r_cuadrado_4 = funcion(lambda p: restriccion_cuadrado_4(p), 2, "Restriccion_Cuadrado_4")

# Lista de restricciones de desigualdad para el cuadrado
restricciones_desigualdad_cuadrado = [r_cuadrado_1, r_cuadrado_2, r_cuadrado_3, r_cuadrado_4]

# Definición de la función F(x1, x2)
def F(x1, x2):
    best_dist = float('inf')
    for x3 in my_linspace(-1, 1, 100):  
        for x4 in my_linspace(-1, 1, 100):
            p = (x3, x4)
            if all(r.f(p) >= 0 for r in restricciones_desigualdad_cuadrado):  # Verificar restricciones
                d = distancia([x1, x2, x3, x4])
                if d < best_dist:
                    best_dist = d
    return best_dist

# Ejecución de la optimización
def optimizar_en_circulo():
    opt_circle = CircleOptimization(lambda x, y: F(x, y), lambda p: restriccion_circulo(p))
    x0, y0 = centro_circulo  # Punto inicial del círculo
    x_opt, y_opt, path = opt_circle.gdescent_con_restriccion(x0, y0)
    best_dist = F(x_opt, y_opt)
    return (x_opt, y_opt), best_dist, path

# Ejecutar la optimización
best_point, min_dist, path = optimizar_en_circulo()
print("La mínima distancia es:", min_dist)
print("El mejor punto es:", best_point)
