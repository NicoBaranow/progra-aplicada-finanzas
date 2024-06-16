import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
sigma1_squared = 0.04  # Varianza del activo 1
sigma2_squared = 0.09  # Varianza del activo 2
covariance = 0.02      # Covarianza entre los activos

# Pesos del activo 1 (W1)
W1 = np.linspace(0, 1, 100)
W2 = 1 - W1  # Pesos del activo 2 (W2)

# Cálculo de la varianza del portafolio para cada peso
portfolio_variance = W1**2 * sigma1_squared + W2**2 * sigma2_squared + 2 * W1 * W2 * covariance

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(W1, portfolio_variance, label='Varianza del Portafolio')
plt.xlabel('Peso del Activo 1 (W1)')
plt.ylabel('Varianza del Portafolio')
plt.title('Varianza del Portafolio en función del Peso del Activo 1')
plt.legend()
plt.grid(True)
plt.show()
