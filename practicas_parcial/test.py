# La inversa por la matriz de resultados me da los valores de x e y. 

#autovalores
import sys
sys.path.append('TP2-Polinomios')
sys.path.append('TP4 - Matrices')


from polis import poly
from matrices import matrix

class autovalores(poly): 
    def __init__(self, rows, cols): 
        self.r = rows
        self.c = cols

def polycar(self):

    for i in range(len(self.elems)):
        self.elems[i] = poly(0,[self.elems[i]])
        


    return self



print(polycar())

mat = matrix([1,2,3,4,5,6,7,2,3],3,3)

new_mat = polycar(mat)
            


        
        

b = matrix([9,5,2,3,4,5,2,4,5],3,3)
poli = poly(1,[0, 1])


identidad = b.identity(3)





