class matrix:
    def __init__ (self, elems = [], rows = 0, columns = 0, by_row = True):
        '''
        elems es una lista con los elementos de la matriz, no una lista de listas. Rows y columns indica la cantidad de filas y columnas
        '''
        if len(elems) != rows*columns: raise ValueError ("Se deben proporcionar igual cantidad de elementos como espacios a llenar en la matriz")
        self.elems = elems
        self.r = rows
        self.c = columns
        self.by_row = by_row

    def __str__(self):
        expression = ""
        for i in range(self.r):
            expression += '[ '
            for j in range(self.c):            
                if self.by_row:
                     expression += str(self.elems[i * self.c + j]) + " "
                else:
                     expression += str(self.elems[j * self.r + i]) + " "
            expression += "] \n"
        return expression
    
    def __mul__ (self,other):
        if self.c != other.r: raise ValueError("Las columnas de la primer matriz deben coincidir con las filas de la segunda")
        
        matrixa = matrix(self.elems,self.r,self.c,self.by_row)
        matrixb = matrix(other.elems,other.r,other.c,other.by_row)
        
        if isinstance(other,(int,float)): return matrix([other * element for element in matrixa],matrixa.r,matrixa.c,matrixa.by_row)
        
        if not matrixa.by_row: matrixa.switch()
        if not matrixb: matrixb.switch()

        result_elems = []
        for i in range(1, matrixa.r + 1):
            for j in range(1, matrixb.c + 1):
                elem = sum(matrixa.get_elem(i, k) * matrixb.get_elem(k, j) for k in range(1, matrixa.c + 1))
                result_elems.append(elem)

        
        return matrix(result_elems, matrixa.r, matrixb.c)

    def switch(self):

        aux= []
        if self.by_row:
            coord_1= self.c
            coord_2= self.r
            self.by_row= False
        else:
            coord_1= self.r
            coord_2= self.c
            self.by_row= True

        for i in range(coord_1):
            for t in range(coord_2):
                aux.append(self.elems[coord_1*t+i])
        
        self.elems= aux

    def identity(self,m,): 
        
        ident = matrix([0]*(m*m),m,m)
        for n in range(1, m+1): ident.elems[ident.get_pos(n,n)] = 1
        return ident
        
    def get_pos(self,j,k):
        '''
        Coordenadas j,k en la matriz del elemento X sub j,k. Devuelve la posicion i en la lista de elementos de la matriz
        '''
        if j > self.r or k > self.c: raise ValueError("Las coordenadas j, k deben estar dentro de la matriz")
        if self.by_row: return (j - 1) * self.c + (k - 1)
        return (k - 1) * self.r + (j - 1)
    
    def get_coords(self,i):
        '''
        Toma la posicion i en la lista de elementos de la matriz y devuelve una tupla (j,k) con la posicion en la matriz
        '''
        if i >= len(self.elems): raise ValueError ("El indice a buscar esta fuera del rango de los elementos de la matriz")
        if self.by_row: return i // self.c + 1, i % self.c + 1
        return i % self.r + 1, i // self.r + 1
        
    def get_row(self,j):
        '''
        Devuelve el contenido de una fila j
        '''
        if j <= 0 or j > self.r: raise ValueError("Índice de fila fuera de rango")
        
        if self.by_row: return self.elems[(j-1) * self.c: j * self.c]
        return self.elems[j - 1: self.c *self.r: self.r]

    def get_col(self,k):
        '''
        Devuelve el contenido de una columna k
        '''
        if k <= 0 or k > self.c: raise ValueError("Índice de fila fuera de rango")
        if not self.by_row: return self.elems[(k-1) * self.r: k * self.r]
        return self.elems[k - 1: self.r * self.c: self.c]

    def get_elem(self,j,k): 
        '''
        Devuelve el elemento (j,k)
        '''
        if j not in range(1,self.r+1): raise ValueError(f"El numero de filas es menor a 1 o mayor a {self.r}")
        if k not in range(1,self.c+1): raise ValueError(f"El numero de columnas es menor a 1 o mayor a {self.k}")

        return self.elems[self.get_pos(j,k)]

    def get_submatrix(self, row_list, col_list):
        submatrix_elems = []
        for row in row_list:
            for col in col_list:
                submatrix_elems.append(self.get_elem(row, col))
        return matrix(submatrix_elems, len(row_list), len(col_list))

    def del_row(self,j):
        '''
        Toma en numero de fila a eliminar, devuelve una nueva matriz sin la fila j
        '''
        ident = self.identity(self.c)
        ident.elems[ident.get_pos(j,j)] = 0
        new_matrix = ident * self
        return matrix(new_matrix.elems[:(j-1)*self.c]+new_matrix.elems[j*self.c:], self.r - 1, self.c)
    
    def del_col(self,j):
        '''
        Toma en numero de columna a eliminar, devuelve una nueva matriz sin la columna j
        '''
        ident = self.identity(self.c)
        ident.elems[ident.get_pos(j,j)] = 0
        new_matrix = self * ident
        
        indices = list(range(j-1, self.r * self.c, self.c))
        nueva_lista = [new_matrix.elems[i] for i in range(self.c * self.r) if i not in indices]
        return matrix(nueva_lista,self.r,self.c-1)

a = matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 4, 4, False)
b = matrix([9,5,2,3,4,5],2,3)

print(a)
print(a.del_col(2))


# funciones del_row( self; j), del_col(self,k) que devuelvan un objeto de la
# clase habiendo eliminando la Öla j o la columna k. Debe presentar dos ver-
# siones de estas funciones siguiendo la lÛgica expuesta en get_row(self; j)
# y get_col (self; k)
# 1


#Segunda version de get_row, get_col, get_elem. "La segunda que sea valiendose de la multiplicacion a izquierda o a derecha por un vector"
# [ 1 2 3 4 ] 
# [ 5 6 7 8 ] 
# [ 9 10 11 12 ] 
# [ 13 14 15 16 ] 