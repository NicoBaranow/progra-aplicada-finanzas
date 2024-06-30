# Importar el manejo de matrices desde matrices_suggested.py
from matrices_suggested import myarray

# Parámetros del problema
M = 1000000  # Total de dinero para invertir
c = [0.01, 0.10, 0.05]  # Costos de transacción para ACN.N, AMZN.O y AAPL.O
mu = [0.12, 0.15, 0.10]  # Rendimientos esperados
Sigma = [
    [0.1, 0.01, 0.02], 
    [0.01, 0.15, 0.01], 
    [0.02, 0.01, 0.2]
]  # Matriz de covarianza
lambda_ = 0.5  # Coeficiente de aversión al riesgo

# Convertir todos los valores de Sigma a float
Sigma = [[float(value) for value in row] for row in Sigma]

# Función para calcular los pesos ajustados por costos de transacción
def calcular_pesos(f):
    total = sum(f[i] * (1 - c[i]) for i in range(len(f)))
    return [f[i] * (1 - c[i]) / total for i in range(len(f))]

# Función objetivo
def F(f):
    w = calcular_pesos(f)
    mu_w = sum(w[i] * mu[i] for i in range(len(w)))
    
    # Convertir w a myarray y hacer rprod con Sigma
    w_array = myarray([float(value) for value in w], len(w), 1, by_row=True)
    Sigma_array = myarray([item for sublist in Sigma for item in sublist], len(Sigma), len(Sigma[0]), by_row=True)
    Sigma_w = Sigma_array.rprod(w_array)
    
    var_w = sum(w[i] * Sigma_w.elems[i] for i in range(len(w)))
    return - (mu_w - lambda_ * var_w)

# Gradiente de la función objetivo
def grad_F(f, h=1e-5):
    grad = []
    for i in range(len(f)):
        f_plus_h = f[:]
        f_plus_h[i] += h
        grad_i = (F(f_plus_h) - F(f)) / h
        grad.append(grad_i)
    return grad

# Restricciones
def restricciones(f):
    if sum(f) != 1:
        return False
    if any(x < 0 for x in f):
        return False
    return True

# Método de descenso por gradiente
def gradient_descent(F, grad_F, f0, alpha=0.01, tol=1e-6, max_iter=10000):
    f = f0[:]
    for _ in range(max_iter):
        grad = grad_F(f)
        f_new = [f[i] - alpha * grad[i] for i in range(len(f))]
        
        # Proyección en el simplex
        suma = sum(f_new)
        if suma != 1:
            f_new = [x / suma for x in f_new]
        
        if restricciones(f_new):
            f = f_new
        else:
            break
        
        if sum(abs(grad_i) for grad_i in grad) < tol:
            break
    return f

# Ejecución de la optimización
def optimizar():
    f0 = [1/3, 1/3, 1/3]
    f_opt = gradient_descent(F, grad_F, f0)
    w_opt = calcular_pesos(f_opt)
    M_opt = [fi * M for fi in f_opt]
    M_inv = [mi * (1 - ci) for mi, ci in zip(M_opt, c)]
    return f_opt, w_opt, M_opt, M_inv

# Ejecutar la optimización
f_opt, w_opt, M_opt, M_inv = optimizar()

print("Solución óptima f:", f_opt)
print("Pesos ajustados w:", w_opt)
print("Montos asignados M_i:", M_opt)
print("Montos invertidos después de costos de transacción:", M_inv)
