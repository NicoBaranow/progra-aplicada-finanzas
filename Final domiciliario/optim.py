# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:36:36 2024

@author: gbasa
"""
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
        
class optim():
    """ Clase que contiene
    
    funcion objetivo: f 
    lista de funciones con restriccion de igualdad: h(x)=0
    lista de funciones con restriccion de desigualdad : g(x)>=0
    
    todas las funciones deben coincidir en el self.nd ( estan todas en el mismo espacio )
    
    Notar que todavia no se determina si el objetivo es de minimizacion o de maximizacion"""
    
    def __init__(self,f,h=[],g=[]): 
        # por default la clase inicializa problemas sin restricciones
        self.f = f
        self.h = h
        self.g = g
        self.minimize = True
        self.coef = -1.0
        self.learn = 0.1
        self.x = None
        
    def solve(self,x0,minimize=True,learning_rate=0.1, tol =0.001):
        self.x = x0
        self.learn = learning_rate 
        
        self.minimize=minimize
        if not(self.minimize): self.coef = 1.0 
            
        condition = True
        
        while condition : 
            w = self.f.grad(self.x)
            xp = [self.x[i] + self.coef * self.learn *  w[i] for i in range(self.f.nd)]   
            delta = sum([(xp[i]-self.x[i])**2.0 for i in range(self.f.nd)])**0.5
            condition = delta> tol
            
            print(xp,delta)
            self.x=xp
            
        return None    
        
import numpy as np

# Función objetivo
def objective_function(x):
    x1, x2, x3, x4 = x
    return np.sqrt((x1 - x3)**2 + (x2 - x4)**2)

# Restricción del círculo
def circle_constraint(x):
    x1, x2 = x
    return 1 - ((x1 - 3)**2 + x2**2)

# Restricciones del cuadrado
def square_constraint_1(x):
    x3, x4 = x
    return 1 - x3 - x4

def square_constraint_2(x):
    x3, x4 = x
    return x4 - x3 + 1

def square_constraint_3(x):
    x3, x4 = x
    return 1 + x3 - x4

def square_constraint_4(x):
    x3, x4 = x
    return x3 + x4 + 1

# Crear instancias de las restricciones y la función objetivo utilizando la clase funcion
f_obj = funcion(objective_function, 4, "Distancia Euclidiana")
f_circle = funcion(circle_constraint, 2, "Restricción Círculo")
f_square_1 = funcion(square_constraint_1, 2, "Restricción Cuadrado 1")
f_square_2 = funcion(square_constraint_2, 2, "Restricción Cuadrado 2")
f_square_3 = funcion(square_constraint_3, 2, "Restricción Cuadrado 3")
f_square_4 = funcion(square_constraint_4, 2, "Restricción Cuadrado 4")

# Crear la instancia de la clase optim
optimizer = optim(f_obj, [f_circle], [f_square_1, f_square_2, f_square_3, f_square_4])

# Definir el punto inicial
x0 = [3.0, 0.0, 0.0, 0.0]  # Punto inicial

# Resolver el problema de optimización
result = optimizer.solve(x0)

if result is not None:
    print(f"El punto óptimo en el círculo es: ({result[0]}, {result[1]})")
    print(f"El punto más cercano en el cuadrado es: ({result[2]}, {result[3]})")
else:
    print("No se encontró una solución.")