#autovalores
import sys
sys.path.append('TP2-Polinomios')
sys.path.append('TP4 - Matrices')


from polis import poly

class matrix(poly):
    def __init__ (self, elems = [], rows = 0, columns = 0, by_row = True):
        '''
        elems es una lista con los elementos de la matriz, no una lista de listas. 
        Rows y columns indica la cantidad de filas y columnas
        by_row indica si se llena fila por fila o colmna por colummna
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
    
    def __mul__(self, mult): #arreglar para cuando la otra matriz False
        new_elems = []
        if isinstance(mult, int) or isinstance(mult,float):
            for i in range(len(self.elems)):
                new_elems.append(self.elems[i]*mult)            
            salida = matrix(new_elems, self.r, self.c)
            
        else:
            if self.columns != mult.rows:
                raise ValueError('the number of rows of array A must be equal to the number of columns in array B')
            for i in range(self.r):
                fila = self.GetRow(i+1)
                for w in range(mult.c):
                    col = mult.GetCol(w+1)
                    suma = 0
                    for x in range(len(fila)):
                        suma += (fila[x]*col[x])
                    new_elems.append(suma)                    
            salida = matrix(new_elems, self.r, mult.c) 
        return salida
    
    def __rmul__(self, mult):
        return self.__mul__(mult)

    # def __mul__ (self,other):
    #     if self.c != other.r: raise ValueError("Las columnas de la primer matriz deben coincidir con las filas de la segunda")
        
        
    #     if isinstance(other,(int,float)): return matrix([other * element for element in self],self.r,self.c,self.by_row)
    #     matrixa = matrix(self.elems,self.r,self.c,self.by_row)
    #     matrixb = matrix(other.elems,other.r,other.c,other.by_row)
        
    #     if not matrixa.by_row: matrixa.switch()
    #     if not matrixb: matrixb.switch()

    #     result_elems = []
    #     for i in range(1, matrixa.r + 1):
    #         for j in range(1, matrixb.c + 1):
    #             elem = sum(matrixa.get_elem(i, k) * matrixb.get_elem(k, j) for k in range(1, matrixa.c + 1))
    #             result_elems.append(elem)
        
    #     return matrix(result_elems, matrixa.r, matrixb.c)

    def __add__(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(self.elems[i]+ B)
        else:
            if (self.r,self.c)!=(B.r,B.c):
                print('No se puede realizar la operacion')
            if self.by_row!=B.by_row:
                B.switch()
            
            for i in range(len(self.elems)):
                aux.append(self.elems[i]+B.elems[i])
                
        res= matrix(aux,self.r,self.c,self.by_row)
        return res
    
    def __sub__(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(self.elems[i]- B)
        else:
            if (self.r,self.c)!=(B.r,B.c):
                print('No se puede realizar la operacion')
            if self.by_row!=B.by_row:
                B.switch()
            
            for i in range(len(self.elems)):
                aux.append(self.elems[i]-B.elems[i])
                
        res= matrix(aux,self.r,self.c,self.by_row)
        return res        

    def rprod(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(B*self.elems[i])
            res= matrix(aux, self.r, self.c, self.by_row)
        else:
            row= []
            col= []
            if self.by_row!=B.by_row:
                B.switch()
            if self.c!=B.r:
                return print('No se puede realizar la operacion')
            for i in range(self.r):
                row= self.get_row(i+1)
                for j in range(B.c):
                    col= B.get_col(j+1)
                    aux.append(sum([row[t]*col[t] for t in range(self.c)]))
                    
            res= matrix(aux,self.r,B.c,self.by_row)    
        return res           
    
    def lprod(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(B*self.elems[i])
            res= matrix(aux, self.r, self.c, self.by_row)
        else:
            res= B.rprod(self)
        return res
    
    def power(self,n):
        for i in range(n-1):
            res= self.rprod(self)
        return res

    def switch(self):
        '''
        Modifica la matriz actual para cambiar el orden de sus elementos internos, para poder operar con una matriz by_row = False como si fuese by_row = True
        '''
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
        '''
        Devuelve una matriz identidad de tamaño m
        '''
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
        '''
        Toma una lista de filas y otra de columnas a interponer
        Devuelve una nueva matriz con la interseccion de filas y columnas correspondiente
        '''

        if any(row not in range(1, self.r) for row in row_list) or any(col not in range(1, self.c+1) for col in col_list): raise ValueError ("Las filas y columnas a interponer deben estar en la matriz")

        submatrix_elems = []
        for row in row_list:
            for col in col_list:
                submatrix_elems.append(self.get_elem(row, col))
        return matrix(submatrix_elems, len(row_list), len(col_list))

    def del_row(self,j):
        '''
        Toma en numero de fila a eliminar, devuelve una nueva matriz sin la fila j
        '''

        if j not in range(1, self.r+1): raise ValueError ("La fila a eliminar debe ser parte de la matriz") 

        ident = self.identity(self.c)
        ident.elems[ident.get_pos(j,j)] = 0
        new_matrix = ident * self
        return matrix(new_matrix.elems[:(j-1)*self.c]+new_matrix.elems[j*self.c:], self.r - 1, self.c)
    
    def del_col(self,j):
        '''
        Toma en numero de columna a eliminar, devuelve una nueva matriz sin la columna j
        '''

        if j not in range(1, self.c+1): raise ValueError ("La columna a eliminar debe ser parte de la matriz") 

        ident = self.identity(self.c)
        ident.elems[ident.get_pos(j,j)] = 0
        new_matrix = self * ident
        
        indices = list(range(j-1, self.r * self.c, self.c))
        nueva_lista = [new_matrix.elems[i] for i in range(self.c * self.r) if i not in indices]
        return matrix(nueva_lista,self.r,self.c-1)

    def swap_cols_index(self,j,k):
        '''
        Toma el número de dos columnas y devuelve una nueva matriz con las columnas intercambiadas.
        '''
        if j not in range(1, self.c+1) or k not in range(1, self.c+1): raise ValueError ("Las columnas a intercambiar deben pertenecer a la matriz")
        
        new_matrix = matrix(self.elems,self.r,self.c,self.by_row)
        if new_matrix.by_row: new_matrix.switch()

        cols = []
        for i in range(new_matrix.c): cols += [new_matrix.get_col(i+1)]
        cols[j-1], cols[k-1] = cols[k-1], cols[j-1]

        return matrix([elemento for sublista in cols for elemento in sublista],self.r,self.c,new_matrix.by_row)
            
    def swap_rows_index(self, j, k):
        '''
        Toma el número de dos filas y devuelve una nueva matriz con las filas intercambiadas.
        '''
        if j not in range(1, self.r+1) or k not in range(1, self.r+1): raise ValueError ("Las filas a intercambiar deben pertenecer a la matriz")
        
        new_matrix = matrix(self.elems,self.r,self.c,self.by_row)
        if not new_matrix.by_row: new_matrix.switch()

        
        rows = []
        for i in range(self.r): rows += [new_matrix.get_row(i+1)]
        rows[j-1], rows[k-1] = rows[k-1], rows[j-1]
        elems = [elemento for sublista in rows for elemento in sublista]
        
        return matrix(elems,self.r,self.c,True)
    
    def scale_row(self, j, x):
        '''
        Toma una fila j y la multiplica por el factor x
        Devuelve una nueva matriz con la fila multiplicada por el factor x
        '''
        if not 1 <= j <= self.r: raise ValueError("La fila j debe estar dentro de la matriz")
        
        new_matrix = matrix(self.elems, self.r, self.c, self.by_row)
        if not self.by_row: new_matrix.switch()

        for i in range((j - 1) * self.c, (j - 1) * self.c + self.c): new_matrix.elems[i] *= x

        if not self.by_row: new_matrix.switch()

        return new_matrix

    def scale_col(self, k, y):
        '''
        Toma una columna k y la multiplica por el factor y
        Devuelve una nueva matriz con la columna multiplicada por el factor y
        '''
        if not 1 <= k <= self.c: raise ValueError("La columna k debe estar dentro de la matriz")

        new_matrix = matrix(self.elems, self.r, self.c, self.by_row)
        if not self.by_row: new_matrix.switch()
        
        for i in range(k - 1, self.r * self.c, self.c): new_matrix.elems[i] *= y

        if self.by_row: new_matrix.switch()

        return new_matrix

    def transpose(self):
        '''
        Devuelve una nueva matriz trasponiendo la original
        '''       
        new_matrix = matrix(self.elems, self.r, self.c, self.by_row)
        if not self.by_row: new_matrix.switch()

        new_elems = []

        for col in range(self.c): new_elems.extend(self.get_col(col + 1))

        new_matrix = matrix(new_elems, self.c, self.r)
        if not self.by_row: new_matrix.switch()
        
        return new_matrix

    def flip_cols(self):
        '''
        Devuelve una nueva matriz con las columnas invertidas
        '''
        new_matrix = matrix(self.elems, self.r, self.c, self.by_row)
        if self.by_row: new_matrix.switch()

        cols = []
        for i in range(self.c): cols.extend(new_matrix.get_col(self.c - i))

        new_matrix = matrix(cols, self.r, self.c,False)
        if not self.by_row: new_matrix.switch()

        return new_matrix
               
    def flip_rows(self):
        '''
        Devuelve una nueva matriz con las filas invertidas
        '''
        new_matrix = matrix(self.elems, self.r, self.c, self.by_row)
        if not self.by_row: new_matrix.switch()

        rows = []
        for i in range(self.r): rows.extend(new_matrix.get_row(self.r - i))

        new_matrix = matrix(rows, self.r, self.c)
        if not self.by_row: new_matrix.switch()

        return new_matrix

    def det(self):
        '''
        Devuelve el determinante de la matriz
        '''
        if self.r!=self.c: raise ValueError('No es una matriz cuadrada')
        
        i=1
        det=0
        def sub_det(i,j):
            M_sub= self.del_row(i)
            M_sub= M_sub.del_col(j)
            return M_sub
            
        if len(self.elems)==1: return self.elems[0]
        
        for j in range(1,self.c+1): det+= self.elems[self.get_pos(i,j)]*((-1)**(i+j))* sub_det(i,j).det()
        return det

    def Minverse(self):
        if self.det()==0 or self.r!= self.c:
            raise ValueError ("No se puede calcular la inversa de esta matriz.")
        
        else:
            A = matrix(self.elems, self.r, self.c)
            inversa = self.identity(self.r)
            for i in range(1,self.c+1):
                columna = A.get_col(i)
                pivot = columna[i-1]
                col = matrix(columna, self.r, 1)
                col_alterada = col.lprod(-1/pivot)
                col_alterada.elems[i-1]=1/pivot
                # print(col_alterada.elems)
                primer_element_de_col = i-1
                pos_col = 0
                identidad_alterada = A.identity(self.r)
                for pos, element in enumerate(identidad_alterada.elems):
                    # identidad = self.matriz_identidad(self.r, self.r)
                    if primer_element_de_col == pos:
                        identidad_alterada.elems[pos] = col_alterada.elems[pos_col]
                        primer_element_de_col += self.c
                        pos_col+=1
                A = A.lprod(identidad_alterada)
                # print("identidad alterada:")
                # print(A.elems)
                inversa = inversa.lprod(identidad_alterada)
                # print("inversa:")
                # print(inversa.elems)
            print(inversa.elems)
            return inversa

    def polycar(self):
        for i in range(len(self.elems)):
            self.elems[i] = poly(0,[self.elems[i]])

        id = self.identity(self.r)
        for i in range(len(id.elems)):
            self.elems[i] = self.elems[i]* poly(1,[0,1])

        px = self - id 


        return px

a = matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 2, 8, False)
b = matrix([9,5,2,3,4,5,2,4,5],3,3)
print(b.polycar())
# print(a)
# print(a.flip_cols().flip_cols())
# print(b.Minverse())