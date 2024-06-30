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