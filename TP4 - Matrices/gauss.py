from matrices import matrix

class gauss(matrix):
    def __init__(self,lista_A=None,N=0,by_row=True,lista_b=None):
        if lista_A is None:
            print('La lista de elementos de A es vacia')
        else:    
            if len(lista_A)==N*N: 
                self.A = matrix(lista_A,N,N,by_row)
            else:
                print('Inconsistencia entre la lista de elementos ingresada y la dimension de la matriz')
        
        if lista_b is None: 
            self.b = matrix([0]*N,N,1,True)
        else:
            if len(lista_b)==N:
                self.b = matrix(lista_b,N,1,True)
            else:
                print('El termino independiente deberia ser de longitud ',N,' la lista b es de longitud ',len(lista_b))
        
        super(gauss,self).__init__(lista_A,N,N,by_row)
        
        # self.L,self.U= self.factor_LU()
        self.A_1= self.Minverse()
        self.by_row= by_row
        
    def solve_gauss(self,b=None):
        if b==None:
            b= self.b
        else:
            if len(b)==self.r:
                self.b = matrix(b,self.r,1,True)
            else:
                print('El termino independiente deberia ser de longitud ',self.r,' la lista b es de longitud ',len(b))            
        
        res= self.A_1.rprod(b)
        
        return res
        
    def solve_LU(self,b=None):
        if b==None:
            b= self.b
        else:
            if len(b)==self.r:
                self.b = matrix(b,self.r,1,True)
            else:
                print('El termino independiente deberia ser de longitud ',self.r,' la lista b es de longitud ',len(b))            
        L,U= self.factor_LU()
        L_1= L.Minverse()
        U_1= U.Minverse()
        y= L_1.rprod(b)
        x= U_1.rprod(y)
        
        return x
        
    def inv(self):
        print(self.A_1.elems)
        if self.by_row:
            print('El orden es por filas')
        else:
            print('El orden es por columna')
        
        return None
    
    def LU(self):
        print('Matriz A se puede descomponer como multiplicacion de:\n','L=',self.L.elems,'\ny\n','U=',self.U.elems)
        if self.by_row:
            print('El orden es por filas')
        else:
            print('El orden es por columna')
        
        return None
