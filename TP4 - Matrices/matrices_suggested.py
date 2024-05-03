#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:21:28 2024

@author: santiagokestler
"""
class myarray():
    def __init__(self,lista,r,c,by_row):
        self.elems=lista
        self.r= r
        self.c= c
        self.by_row= by_row
        
    def get_pos(self,j,k):
        if j>self.r or k>self.c:
            raise ValueError('Los valores ingresados no coinciden con las dimensiones de la matriz')
        if self.by_row:
            pos= self.c*(j-1)+k-1
        else:
            pos= self.r*(k-1)+j-1
        
        return pos
        
    def get_coords(self,m):
        salir= False
        for j in range(1,self.r+1):
            for k in range(1,self.c+1):
                pos= self.get_pos(j,k)
                
                if pos==m:
                    salir= True
                    break
            if salir:
                break
        
        if self.by_row:
            coords= [j,k]
        else:
            coords= [k,j]
        
        return coords
    
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
        
    def get_row(self,j):
        row= []
        for i in range(1,self.c+1):
            row.append(self.elems[self.get_pos(j,i)])
        return row
                
    def get_col(self,k):
        col= []
        
        if self.by_row:
            for i in range(self.r):
                col.append(self.elems[self.c*i+(k-1)])
        else:
            for i in range(self.c):
                col.append(self.elems[self.r*(k-1)+i])
        
        return col
    
    def get_elem(self,coord):
        elem= self.elems[self.get_pos(coord[0],coord[1])-1]
        
        return elem
    
    def get_submatrix(self,row,col):      
        # DEVUELVE UNA MATRIZ CON LAS FILAS Y COLUMNAS QUE SE INDICARON ELIMINADAS
        aux= []
        sub_elems= self.elems.copy()
        
        for i in range(len(row)):
            for t in range(1,self.c+1):
                aux.append(self.get_pos(row[i],t))
 
        for i in range(len(col)):
            for t in range(1,self.r+1):
                aux.append(self.get_pos(t,col[i]))        
        
        sub_elems = [v for i,v in enumerate(sub_elems) if i not in frozenset(aux)]
        new_r= self.r-len(row)
        new_c= self.c-len(col)
        
        return myarray(sub_elems,new_r,new_c,self.by_row)
    
    def del_row(self,j):
        aux= []
        sub_elems= self.elems.copy()

        for t in range(1,self.c+1):
            aux.append(self.get_pos(j,t))
            
        sub_elems = [v for i,v in enumerate(sub_elems) if i not in frozenset(aux)]
        new_r= self.r-1
        
        return myarray(sub_elems,new_r,self.c,self.by_row)
    
    def del_col(self,k):
        aux= []
        sub_elems= self.elems.copy()
        
        for t in range(1,self.r+1):
            aux.append(self.get_pos(t,k)) 
        
        sub_elems = [v for i,v in enumerate(sub_elems) if i not in frozenset(aux)]
        new_c=self.c-1
        
        return myarray(sub_elems,self.r,new_c,self.by_row)
    
    def swap_rows(self,j,k):
        row_j=[]
        row_k=[]
        aux= self.elems.copy()
        
        for i in range(1,self.c+1):
            row_j.append(self.elems[self.get_pos(j,i)])
            row_k.append(self.elems[self.get_pos(k,i)])
            aux[self.get_pos(j,i)]= row_k[i-1]
            aux[self.get_pos(k,i)]= row_j[i-1]
        
        return myarray(aux,self.r,self.c,self.by_row)
    
    def swap_cols(self,l,m):
        col_l=[]
        col_m=[]
        aux= self.elems.copy()
        
        for i in range(1,self.r+1):
            col_l.append(self.elems[self.get_pos(i,l)])
            col_m.append(self.elems[self.get_pos(i,m)])
            aux[self.get_pos(i,l)]= col_m[i-1]
            aux[self.get_pos(i,m)]= col_l[i-1]
        
        return myarray(aux,self.r,self.c,self.by_row)
    
    def scale_row(self,j,x):
        aux= self.elems.copy()
        
        for i in range(1,self.c+1):
            aux[self.get_pos(j,i)] *= x
        
        return myarray(aux,self.r,self.c,self.by_row)
    
    def scale_col(self,k,y):
        aux= self.elems.copy()
        
        for i in range(1,self.r+1):
            aux[self.get_pos(i,k)] *= y
        
        return myarray(aux,self.r,self.c,self.by_row)
    
    def transpose(self):
        aux= []
        for i in range(1,self.c+1):
            for t in range(1,self.r+1):
                aux.append(self.elems[self.get_pos(t,i)])
        
        return myarray(aux,self.c,self.r,self.by_row)
    
    def flip_cols(self):
        aux= []
        if self.by_row:
            for i in range(1,self.r+1):
                for t in reversed(range(1,self.c+1)):
                    aux.append(self.elems[self.get_pos(i,t)])
        else:
            for i in reversed(range(1,self.c+1)):
                for t in range(1,self.r+1):
                    aux.append(self.elems[self.get_pos(i,t)])
                    
        return myarray(aux,self.r,self.c,self.by_row)
    
    def flip_rows(self):
        aux= []
        if self.by_row:
            for i in reversed(range(1,self.r+1)):
                for t in range(1,self.c+1):
                    aux.append(self.elems[self.get_pos(i,t)])
        else:
            for i in range(1,self.c+1):
                for t in reversed(range(1,self.r+1)):
                    aux.append(self.elems[self.get_pos(i,t)])
                    
        return myarray(aux,self.r,self.c,self.by_row)

    def det(self):
        i=1
        det=0
        if self.r!=self.c:
            raise ValueError('No es una matriz cuadrada')
        def sub_det(i,j):
            M_sub= self.del_row(i)
            M_sub= M_sub.del_col(j)
            return M_sub
            
        if len(self.elems)==1:
            return self.elems[0]
        else:
            for j in range(1,self.c+1):
                det+= self.elems[self.get_pos(i,j)]*((-1)**(i+j))* sub_det(i,j).det()
            return det
        
    def add(self,B):
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
                
        res= myarray(aux,self.r,self.c,self.by_row)
        return res
    
    def sub(self,B):
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
                
        res= myarray(aux,self.r,self.c,self.by_row)
        return res        
    
    def rprod(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(B*self.elems[i])
            res= myarray(aux, self.r, self.c, self.by_row)
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
                    
            res= myarray(aux,self.r,B.c,self.by_row)    
        return res           
    
    def lprod(self,B):
        aux= []
        if isinstance(B,int) or isinstance(B,float):
            for i in range(len(self.elems)):
                aux.append(B*self.elems[i])
            res= myarray(aux, self.r, self.c, self.by_row)
        else:
            res= B.rprod(self)
        return res
    
    def power(self,n):
        for i in range(n-1):
            res= self.rprod(self)
        return res
    
    def get_row_v2(self,j):
        M= [0]*self.r
        M[j-1]= 1
        matrix= myarray(M,1,self.r,self.by_row)
        res= self.lprod(matrix)
        
        return res.elems
    
    def get_col_v2(self,k):
        M= [0]*self.c
        M[k-1]= 1
        matrix= myarray(M,self.c,1,self.by_row)
        res= self.rprod(matrix)
        
        return res.elems
        
    def get_elem_v2(self,coord):
        M= [0]*self.c
        M[coord[1]-1]= 1       
        matrix= myarray(M,self.r,1,self.by_row)
        res= self.rprod(matrix)
        
        M_2= [0]*self.r
        M_2[coord[0]-1]= 1
        matrix_2= myarray(M_2,1,self.c,self.by_row)
        elem= res.lprod(matrix_2)
        
        return elem.elems[0]
    
    def del_row_v2(self,j):
        M= [0]*(self.r-1)*self.r
        M= myarray(M,self.r-1,self.r,self.by_row)
        
        for i in range(1,M.r+1):
            for t in range(1,M.c+1):
                if j==1:
                    if t==(i+1): M.elems[M.get_pos(i,t)]= 1                    
                if j<M.r and j!=1:
                    if i==t and i<j: M.elems[M.get_pos(i,t)]= 1
                    if i==j and t==(i+1): M.elems[M.get_pos(i,t)]= 1
                    if t==(i+1) and i>j: M.elems[M.get_pos(i,t)]= 1
                if j>=M.r and j!=1:
                    if i==t and i!=j: M.elems[M.get_pos(i,t)]= 1
                    if i==j and t==(j+1): M.elems[M.get_pos(i,t)]= 1
        
        res= self.lprod(M)
        return res                   
                    
    def del_col_v2(self,k):
        M= [0]*self.c*(self.c-1)
        M= myarray(M,self.c,self.c-1,self.by_row)
        
        for i in range(1,M.c+1):
            for t in range(1,M.r+1):
                if k==1:
                    if t==(i+1): M.elems[M.get_pos(t,i)]= 1                    
                if k<M.c and k!=1:
                    if i==t and i<k: M.elems[M.get_pos(t,i)]= 1
                    if i==k and t==(i+1): M.elems[M.get_pos(t,i)]= 1
                    if t==(i+1) and i>k: M.elems[M.get_pos(i,t)]= 1
                if k>=M.c and k!=1:
                    if i==t and i!=k: M.elems[M.get_pos(t,i)]= 1
                    if i==k and t==(k+1): M.elems[M.get_pos(t,i)]= 1
        
        res= self.rprod(M)
        return res     
    
    def swap_rows_v2(self,j,k):
        P=[0]*(self.r**2)
        P= myarray(P,self.r,self.r,self.by_row)
        P.elems[P.get_pos(j,k)]= 1
        P.elems[P.get_pos(k,j)]= 1
        
        for i in range(1,self.r+1):
            if i!=j and i!=k:
                P.elems[P.get_pos(i,i)]= 1
        
        res= self.lprod(P)
        
        return res,P
        
    def swap_cols_v2(self,l,m):
        P=[0]*(self.c**2)
        P= myarray(P,self.c,self.c,self.by_row)
        P.elems[P.get_pos(l,m)]= 1
        P.elems[P.get_pos(m,l)]= 1
        
        for i in range(1,self.c+1):
            if i!=l and i!=m:
                P.elems[P.get_pos(i,i)]= 1
        
        res= self.rprod(P)
        
        return res,P        
    
    def scale_row_v2(self,j,x):
        S=[0]*(self.r**2)
        S= myarray(S,self.r,self.r,self.by_row)
        S.elems[S.get_pos(j,j)]= x
        
        for i in range(1,self.r+1):
            if i!=j:
                S.elems[S.get_pos(i,i)]= 1
        
        res= self.lprod(S)
        
        return res
    
    def scale_col_v2(self,k,y):
        S=[0]*(self.c**2)
        S= myarray(S,self.c,self.c,self.by_row)
        S.elems[S.get_pos(k,k)]= y
        
        for i in range(1,self.c+1):
            if i!=k:
                S.elems[S.get_pos(i,i)]= 1
        
        res= self.rprod(S)
        
        return res
    
    def I(self):
        I= [0]*(self.r**2)
        I= myarray(I,self.r,self.r,self.by_row)
        
        for i in range(1,self.r+1):
            I.elems[I.get_pos(i,i)]= 1
        
        return I
    
    def Gauss(self):
        if self.r!=self.c:
            raise ValueError('la matriz no es cuadrada')
        if self.det()==0:
            raise ValueError('El determinante es 0')
        
        A_p= self.lprod(self.I())
        L= []
        Perm= []
        
        for i in range(1,self.r+1):
            if i<self.r:
                buscar= A_p.get_col(i)
                for j in range(i,self.r+1):
                    if abs(buscar[j-1])>abs(buscar[i-1]):
                        break
                if buscar[j-1]!=0:
                    A_p,P= A_p.swap_rows_v2(i,j) 
                
            pivot= A_p.get_col(i)
            pivot_value= pivot[i-1]
            pivot= [(-1)*pivot[m] for m in range(len(pivot))]
            pivot[i-1]= 1
        
            factor= [(1/pivot_value)*pivot[m] for m in range(len(pivot))]
            L.append(self.I())
            for l in range(1,L[i-1].r+1):
                L[i-1].elems[L[i-1].get_pos(l,i)]= factor[l-1]
                
            if abs(buscar[j-1])>abs(buscar[i-1]): 
                Perm.append(P)
            else:
                Perm.append(A_p.I())
               
            A_p= A_p.lprod(L[i-1])

        return L,Perm
    
    def Minverse(self):
        L,P= self.Gauss()
        inv= L[0].rprod(P[0])
        
        for i in range(1,len(L)):
            inv= inv.lprod(P[i])
            inv= inv.lprod(L[i])
            
        return inv
    
    def factor_LU(self):
        A_p= self.lprod(self.I())
        L_g= []
        pivot_mat= self.I()
        
        for i in range(1,self.r+1):                
            pivot= A_p.get_col(i)
            pivot_value= pivot[i-1]
            if pivot_value==0:
                break
            pivot_mat.elems[pivot_mat.get_pos(i,i)]= pivot_value
            pivot= [(-1)*pivot[m] for m in range(len(pivot))]
            for x in range(0,i):
                pivot[x]= 0
            pivot[i-1]= 1
        
            factor= [(1/pivot_value)*pivot[m] for m in range(len(pivot))]
            L_g.append(self.I())
            for l in range(1,L_g[i-1].r+1):
                L_g[i-1].elems[L_g[i-1].get_pos(l,i)]= factor[l-1]
               
            A_p= A_p.lprod(L_g[i-1])
            
        L= L_g[0]
        for i in range(1,len(L_g)):
            L= L.lprod(L_g[i])        
        
        U= L.rprod(self)
        L= L.lprod(pivot_mat)
        U= U.lprod(pivot_mat)
        L= L.Minverse()
        
        return L,U
        
        
        

class lineq(myarray):
    def __init__(self,lista_A=None,N=0,by_row=True,lista_b=None):
        if lista_A is None:
            print('La lista de elementos de A es vacia')
        else:    
            if len(lista_A)==N*N: 
                self.A = myarray(lista_A,N,N,by_row)
            else:
                print('Inconsistencia entre la lista de elementos ingresada y la dimension de la matriz')
        
        if lista_b is None: 
            self.b = myarray([0]*N,N,1,True)
        else:
            if len(lista_b)==N:
                self.b = myarray(lista_b,N,1,True)
            else:
                print('El termino independiente deberia ser de longitud ',N,' la lista b es de longitud ',len(lista_b))
        
        super(lineq,self).__init__(lista_A,N,N,by_row)
        
        self.L,self.U= self.factor_LU()
        self.A_1= self.Minverse()
        self.by_row= by_row
        
    def solve_gauss(self,b=None):
        if b==None:
            b= self.b
        else:
            if len(b)==self.r:
                self.b = myarray(b,self.r,1,True)
            else:
                print('El termino independiente deberia ser de longitud ',self.r,' la lista b es de longitud ',len(b))            
        
        res= self.A_1.rprod(b)
        
        return res
        
    def solve_LU(self,b=None):
        if b==None:
            b= self.b
        else:
            if len(b)==self.r:
                self.b = myarray(b,self.r,1,True)
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
        
    
    
    
    
    
    
    
    