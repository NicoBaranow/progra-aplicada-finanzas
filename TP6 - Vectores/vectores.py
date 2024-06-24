"""
Created on Mon Jun 10 12:58:50 2024

@author: gbasaluzzo
"""

class vector:
    def __init__(self,v):
        self.x   = v
        self.eps = 0.00001
        self.elems = v
        
    def inner(self,v): 
        return sum([self.x[i]*v.x[i] for i in range(len(self.x))])
    
    def norm(self):
        return (self.inner(self))**0.5
    
    def versor(self):
        a = self.norm()
        if a < self.eps:  # Verificación para evitar división por cero
            # raise ValueError("Cannot normalize a zero vector")
            return vector([0 for i in range(len(self.x))])
        return vector([self.x[i] / a for i in range(len(self.x))])    
    def orth(self,v):
        return self - v.versor()*self.inner(v.versor())
    
    def __add__(self,v):
        return vector([self.x[i]+v.x[i] for i in range(len(self.x))])

    def __sub__(self,v):
        return vector([self.x[i]-v.x[i] for i in range(len(self.x))])
    
    def __mul__(self,b):
        if isinstance(b,int)  or isinstance(b,float):
            salida = vector([self.x[i]*b for i in range(len(self.x))]) 
        else: 
             salida = None 
        return salida 
    
    def __rmul__(self,b):
        if isinstance(b,int)  or isinstance(b,float):
            salida = vector([self.x[i]*b for i in range(len(self.x))]) 
        else: 
             salida = None 
        return salida 
    
    def minus(self, other = None):
        if other == None: return self
        return vector([a - b for a, b in zip(self.x, other.x)])
    
    def __str__(self):
        salida ="("
        for i in range(len(self.x)-1):
            salida = salida + str(self.x[i])+","
        salida = salida + str(self.x[len(self.x)-1])+")"
        return salida


    def orth_proj(self,lista):
        import sys        
        # Computa la proyeccion ortogonal a los vectores de la lista
        
        # primero armo ortonormalizacion de Gram Schmidt
        N = len(lista)

        lista_norm=[lista[0].versor()]
        if N > 1:
            for i in range(1,len(lista)):
                v = lista[i]
                for j in lista_norm:                    
                    v = v - j.inner(v)*j
                
                if v.norm()<self.eps:
                    sys.exit('En la ortogonalizacion, uno de los vectores no es independiente')
                else:
                    v = v.versor()
                    
                lista_norm.append(v) 
        # Ahora computo la proyeccion ortogonal al conjunto generado por 
        # los vectores de la lista
         
        v = self
        for j in lista_norm:
            v = v - v.inner(j)*j
        return v,lista_norm
            
            
            
                
#%%        
X1 = vector([1,2,3,4])
X2 = vector([-1,2,-1,0]) 
X3 = vector([1,1,0,-1])
X4 = vector([2,0,-1,2])
X5 = vector([4,2,1,1])

v = X2-X1

w1 = X3-X2
w2 = X4-X2
w3 = X5-X2

# Construyo una base ortogonal de R3 --  espacio de una dimension menos que los X --
w1_hat = w1.versor() 
w2_hat =(w2-(w2.inner(w1_hat)*w1_hat )).versor()
w3_hat = (w3-(w3.inner(w1_hat)*w1_hat)-(w3.inner(w2_hat)*w2_hat)).versor()

# El vector perpendicular a todos los versores y por donde deberia ir la reflexion en polytopes es
vp,w = v.orth_proj([w1,w2,w3])


