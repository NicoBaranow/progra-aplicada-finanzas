import matplotlib.pyplot as plt
import numpy as np
import math
import random

class BisectionMethod:
    def find_root(self, func, a = -1_000_000, b = 1_000_000, x_tolerance=1e-5, y_tolerance=1e-5, max_iterations=100000):
        if func(a) * func(b) >= 0:
            raise ValueError("The function must have different signs at the endpoints a and b.")

        for _ in range(max_iterations):
            mid = (a + b) / 2.0  # Compute the midpoint in each iteration
            f_mid = func(mid)
            
            if abs(f_mid) < y_tolerance or (b - a) / 2 < x_tolerance:
                return mid
            if f_mid * func(a) < 0:
                b = mid
            else:
                a = mid

        return None


class NewtonMethod:
    def find_root(self, func, derivative, x0 = 0.01, x_tolerance=1e-5, y_tolerance = 1e-5, max_iterations=100000, x_step = 1_000):
        x = x0
        for i in range(max_iterations):
            f_x = func(x)
            df_x = derivative(x)
            
            if abs(f_x) < y_tolerance:
                return x

            if df_x == 0:
                raise ValueError(f"Derivative is zero at x = {x}, cannot continue Newton's method")
            
            x_new = x - f_x / df_x
            if abs(x_new - x) < x_tolerance:
                return self.find_root(func, derivative, x0 = random.uniform(x0-x_step, x0+x_step), 
                                      x_tolerance = x_tolerance, y_tolerance = y_tolerance, 
                                      max_iterations = max_iterations - i - 1, x_step = x_step)
            x = x_new

        return None

class Polinomios():
    def __init__(self, n=0, coefs = [0]):
        if len(coefs) != n+1:
            raise ValueError('El numero de coeficientes debe ser igual al grado indicado + 1')
        self.grado = n
        self.coefs = coefs

    
    def Get_Expression(self):
        poli_str = ' '        
       
        for i, coef in enumerate(self.coefs[:-1]):
            if abs(coef) >= 1e-5:
                poli_str += (f'{coef}x^{self.grado-i} ')
        
        if self.coefs[-1] != 0:
            poli_str += (f'{self.coefs[-1]}')
        
        return poli_str

    def poly_plt(self, a, b, **kwargs):
        x = np.linspace(a, b, 1000)
        y = np.polyval(self.coefs, x)
        plt.plot(x, y, **kwargs)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Grafico de Polinomio')
        plt.grid(True)
        plt.show()
              
    def __call__(self, x):
        resultado = sum(coef * x**(self.grado - i) for i, coef in enumerate(self.coefs))
        return resultado


    def __add__(self, polinomio):
        
        if isinstance(polinomio, int) or isinstance(polinomio, float):
            suma_coefs = self.coefs[:-1]
            suma_coefs.append(self.coefs[-1] + polinomio)
            maximo_grado = self.grado
                       
        else:
            if len(polinomio.coefs) > len(self.coefs):
                diferencia = len(polinomio.coefs)-len(self.coefs)
                self.coefs = [0]*diferencia + self.coefs
                
            elif len(polinomio.coefs) < len(self.coefs):
                diferencia = len(self.coefs)-len(polinomio.coefs)
                polinomio.coefs = [0]*diferencia +polinomio.coefs
                
            suma_coefs = [coef_p + coef_self for coef_p, coef_self in zip(polinomio.coefs, self.coefs)]
            maximo_grado = max(polinomio.grado, self.grado)
        
        nuevo_poly = Polinomios(maximo_grado, suma_coefs)        
        return nuevo_poly
    
    def __radd__(self, other):
        return self.__add__(other)    
        
    def __sub__(self, polinomio):
        
        if isinstance(polinomio, int) or isinstance(polinomio, float):
            resta_coefs = self.coefs[:-1]
            resta_coefs.append(self.coefs[-1] - polinomio)
            maximo_grado = self.grado
        
        else:   
            if len(polinomio.coefs) > len(self.coefs):
                diferencia = len(polinomio.coefs)-len(self.coefs)
                self.coefs = [0]*diferencia + self.coefs
                
            elif len(polinomio.coefs) < len(self.coefs):
                diferencia = len(self.coefs)-len(polinomio.coefs)
                polinomio.coefs = [0]*diferencia +polinomio.coefs
                
            resta_coefs = [coef_self - coef_p for coef_self, coef_p in zip(self.coefs, polinomio.coefs)]
            maximo_grado = max(polinomio.grado, self.grado)
            
        nuevo_poly = Polinomios(maximo_grado, resta_coefs)        
        return nuevo_poly
    
    def __rsub__(self, other):
        return self.__sub__(other)

    
    def __mul__(self, polinomio):
             
        if isinstance(polinomio, int) or isinstance(polinomio, float):
            mul_coefs = []
            for coef in self.coefs:
                mul_coefs.append(coef * polinomio)
            maximo_grado = self.grado
            
        else:        
            mul_coefs = [0] * (len(self.coefs) + len(polinomio.coefs) - 1)    
            
            for i, coef_self in enumerate(self.coefs):
                for j, coef_p in enumerate(polinomio.coefs):
                    mul_coefs[i + j] += coef_self * coef_p
        
        
            maximo_grado = self.grado + polinomio.grado
        nuevo_poly = Polinomios(maximo_grado, mul_coefs)
        
        return nuevo_poly 
    
    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, polinomio):        
        if isinstance(polinomio, int) or isinstance(polinomio, float):
            if polinomio == 0:
                raise ZeroDivisionError("Division by zero polynomial")
            else:
                cociente = []
                for coef in self.coefs:
                    cociente.append(coef / polinomio)
                maximo_grado = self.grado
                nuevo_poly = Polinomios(maximo_grado, cociente)                            
            
        else:
            if len(polinomio.coefs) == 0:
                raise ZeroDivisionError("Division by zero polynomial")
            
            elif polinomio.grado > self.grado:
                raise ZeroDivisionError("Division is not possible; the degree of the dividend must be greater than the degree of the divisor.")
            
            else:
                dividendo = self.coefs
                divisor = polinomio.coefs
                cociente = [0]*(self.grado - polinomio.grado + 1)

                y = 0
                while len(dividendo) >= len(divisor):                    
                    cociente[y] = dividendo[0] / divisor[0]
                
                    for i in range(len(divisor)):
                        dividendo[i] -= cociente[y] * divisor[i]                            
                    dividendo.pop(0)
                    y +=1
                
                resultado = cociente
                maximo_grado = len(resultado) - 1
                resto = dividendo        
                nuevo_poly = Polinomios(maximo_grado, resultado)                            
        
        return nuevo_poly

    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __floordiv__(self, other):
        quotient = self.__truediv__(other)
        return quotient

    def __rfloordiv__(self, other):
        if isinstance(other, Polinomios):
            return other.__truediv__(self)[0]
        else:
            return Polinomios(n=0, coefs=[other])._divide(self)[0]

    def __mod__(self, polinomio):
        if isinstance(polinomio, int) or isinstance(polinomio, float):
            if polinomio == 0:
                raise ZeroDivisionError("Division by zero polynomial")
            else:
                cociente = []
                for coef in self.coefs:
                    cociente.append(coef * polinomio) 
                
                maximo_grado = len(cociente) - 1                                                   
                nuevo_poly = Polinomios(maximo_grado, cociente)                                                        
            
        else:
            if len(polinomio.coefs) == 0:
                raise ZeroDivisionError("Division by zero polynomial")
            
            elif polinomio.grado > self.grado:
                raise ZeroDivisionError("Division is not possible; the degree of the dividend must be greater than the degree of the divisor.")
            
            else:
                dividendo = self.coefs
                divisor = polinomio.coefs
                cociente = [0]*(self.grado - polinomio.grado + 1)

                y = 0
                while len(dividendo) >= len(divisor):                    
                    cociente[y] = dividendo[0] / divisor[0]
                
                    for i in range(len(divisor)):
                        dividendo[i] -= cociente[y] * divisor[i]                            
                    dividendo.pop(0)
                    y +=1
                                           
                resto = dividendo
                maximo_grado = len(dividendo) - 1
                nuevo_poly = Polinomios(maximo_grado, resto)                                                                     
                
        return nuevo_poly
    
    def __rmod__(self, other):
        return self.__mod__(other)
    
    
    def Derivar(self):
        derivada = []
        
        for i, coef in enumerate(self.coefs[:-1]):
            derivada.append(coef*(self.grado - i))
        
        maximo_grado = len(derivada) - 1
        nuevo_poly = Polinomios(maximo_grado, derivada)
        
        return nuevo_poly
         
    def Rootfind(self, initial_guess=1, max_iterations=1000, tolerance=1e-6):                
        for i in range(max_iterations):
            fx = self(initial_guess)
            if abs(fx) < tolerance:
                return round(initial_guess, 2)

            funcion_derivada = self.Derivar() 
            funcion_derivada_ev = funcion_derivada(initial_guess)
            if funcion_derivada_ev == 0:
                initial_guess += tolerance
                continue
            
            initial_guess -= fx / funcion_derivada_ev  # Corregido
                    
        return initial_guess

    def FindRoots(self, initial_guess=1):        
        raices = []
        maximas_raices = self.grado
        contador = 0
        while contador < maximas_raices:
            raiz = self.Rootfind()
            if self(raiz) == 0 or self(raiz) < 1e-6:
                raices.append(raiz)            
                poli_raiz = Polinomios(1, [1, -raiz])        
                self = self / poli_raiz          
            contador += 1
            
        raices.sort()
        
        salida = []
        for raiz in raices:
            multiplicidad = raices.count(raiz)
            tupla = (raiz, multiplicidad)
            if tupla not in salida:
                salida.append(tupla)
                                               
        return salida
    
    
    def rootfind(self, method='newton', **kwargs):
        if method == 'bisection':
            root_finder = BisectionMethod()
            return root_finder.find_root(self.__call__, **kwargs)
        elif method == 'newton':
            root_finder = NewtonMethod()
            derivative_func = lambda x: self.fprime(1)(x)
            return root_finder.find_root(self.__call__, derivative_func, **kwargs)
        else:
            raise ValueError("Unsupported root finding method.")

    def findroots(self, method = 'newton', **kwargs):
        roots = []
        residual_poly = self
    
        while residual_poly.n > 0:
            root = residual_poly.rootfind(method, **kwargs)        
            if not root:
                break
            
            multiplicity = 0

            while True:
                quotient, remainder = residual_poly._divide(Polinomios(1, [-root, 1]))

                if remainder.n == 0 and abs(remainder.coefs[0]) < 1e-3:
                    multiplicity += 1
                    residual_poly = quotient
                else:
                    break

            if multiplicity > 0:
                roots.append((root, multiplicity))
        
        return roots, residual_poly
    
    def Factorize(self):
        raices = self.FindRoots()
        polinomio_factorizado = ''
        
        for raiz in raices:
            x0 = raiz[0]
            multiplicidad = raiz[1]
            if x0 == 0:
                polinomio_factorizado += f'(x)**{multiplicidad} '
            else:
                if x0 < 0:
                    x0 = abs(x0)
                    polinomio_factorizado += f'(x + {x0})**{multiplicidad} '
                else:
                    polinomio_factorizado += f'(x - {x0})**{multiplicidad} '
        
        print(polinomio_factorizado)
    
    def fprime(self, k, x0=None):
        poly_a_derivar = self
        if k > self.grado:
            salida = 0
        else:
            contador_grado = 0
            while contador_grado < k:
                der_nesima = poly_a_derivar.Derivar()
                poly_a_derivar = der_nesima
                contador_grado += 1
            nuevo_poli = poly_a_derivar
            if x0 == None:
                salida = nuevo_poli
            else:
                salida = nuevo_poli(x0)
        return salida


class LinearPoly(Polinomios):
    def __init__(self, m, b):
        super().__init__(1,[m,b])

    def Derivar(self):
        return LinearPoly(0, self.coefs[0])
    
    def Rootfind(self):
        return -self.coefs[1]/self.coefs[0]
    
    def FindRoots(self):
        return [(self.Rootfind(), 1)]
    
class QuadraticPoly(Polinomios):
    def __init__(self,a,b,c):
        super().__init__(2,[a,b,c])
    
    def Derivar(self):
        d_a = 2*self.coefs[0]
        d_b = self.coefs[1]
        return LinearPoly(1, [d_a, d_b])

    def Rootfind(self):
        a,b,c = self.coefs
        discriminante = b**2 -4*a*c
        if discriminante == 0:
            salida = None
        else:
            raiz_1 = ( ( -b + discriminante**(1/2) ) / (2*a) )
            raiz_2 = ( ( -b - discriminante**(1/2) ) / (2*a) )
            if raiz_1 == raiz_2:
                salida = raiz_1
            else:
                salida = raiz_1, raiz_2

        return salida
    
    def FindRoots(self):
        raices = self.Rootfind()
        if raices == None:
            salida = []
        elif isinstance(raices, tuple):
            salida = [(raices[0], 1), (raices[1], 1)]
        else:
            salida = [(raices, 2)]
        return salida
    
           
        
#polinomio1 = Polinomios(6,[6, 0, 0, -5, 3, 2, -4])
#print(polinomio1.fprime(2,2))

#%%


class Taylor(Polinomios):
    def __init__(self, ft,N,x0,h=0.01, prtTaylor = bool):
        self.funcion = ft
        self.grado = N
        self.x0 = x0 
        self.incremento = h
        self.feval = [self.funcion(self.x0+(self.grado-2j)*self.incremento) for j in (range(self.grado+1))]
        self.fprime = [self.Derivada_n(j) for j in range(self.grado+1)] 
        self.prtTaylor = prtTaylor
        self.digits = 3
        super(Taylor, self).__init__(self.GetParms())
    
            
    def Derivada_n(self, n):        
        def combinatoria(n,i):
            if n == i or i == 0:
                salida = 1
            else:
                salida = combinatoria(n-1, i-1) + combinatoria(n-1, i)                
            return salida            
        sumatoria = 0
        for i in range(n+1):
            sumatoria += (((-1)**i) * (combinatoria(n,i)*(self.feval[i]))/(2*self.incremento)**n)        
        return sumatoria    
    
    def __str__(self):
        if self.prtTaylor:
            terms = []
            for i, coef in enumerate(self.fprime):
                terms.append(f"{round(coef, self.digits)}(x - {self.x0})^{i}/{i}!")
            return 'p(x)= ' + " + ".join(terms)
        else:
            return super().__str__()

    def GetParms(self):        
        def fact(n):
            if n == 1 or n == 0:
                salida = 1
            else:
                salida = fact(n-1)*n
            return salida
            
        aux = [self.fprime(self.grado-j)/fact(self.grado-j) for j in range (self.grado +1)] #coeficientes en forma decreciente
        monomio = Polinomios(1,[1,-self.x0])
        salida = 0
        for i, a_coef in enumerate(aux):
            if i == 0:
                taylor_poly = a_coef
            else:
                taylor_poly += a_coef * monomio**i
            
        return taylor_poly.coefs
        

#%%

class linreg(Polinomios):
    def __init__(self, data):
        self.data = data
        self.xvalues = [tupla[0] for tupla in self.data]
        self.yvalues = [tupla[1] for tupla in self.data]
        
        self.x_hat = sum(self.xvalues) / len(self.xvalues)
        self.y_hat = sum(self.yvalues) / len(self.yvalues)
        
        self.beta_MCO = self.CalculateBeta()
        self.alpha = self.y_hat - self.beta_MCO*self.x_hat
        
        super().__init__(n=1, coefs=[self.alpha,self.beta_MCO])
    
    def __str__(self):
        beta_rounded = round(self.beta_MCO, 2)
        alpha_rounded = round(self.alpha, 2)
        return f"y = {beta_rounded}x + {alpha_rounded}"
    
    def CalculateBeta(self):
        numerador = 0
        denominador = 0
        for i in range(len(self.data)):
            numerador += (self.xvalues[i] - self.x_hat) * (self.yvalues[i] - self.y_hat)
            denominador += (self.xvalues[i] - self.x_hat)**2
        return numerador/denominador
        
    def regplot(self):
        self.yvalues_interpolated = self.Interpolate(self.xvalues)
        
        plt.scatter(self.xvalues,self.yvalues)
        plt.plot(self.xvalues,self.yvalues_interpolated, color = 'red', label = str(self))
        plt.grid()
        plt.legend()
    
    def Interpolate(self, xvalues):
        new_yvalues = [self(x) for x in xvalues]
        return new_yvalues
        
# lr = linreg([(0,1.3),(1,1.8),(2,3.2),(3,3.9)])
# lr.regplot()
       
# nr_reg derivas y newton rapson 

     

#%%

class Lagrange(Polinomios):
    def __init__(self, data):
        self.data = data
        self.x_values = [tupla[0] for tupla in self.data]
        self.y_values = [tupla[1] for tupla in self.data]
        self.lagrange_coefs = self._get_lagrange_coefs()
        self.lagrange_grade = len(self.lagrange_coefs) - 1
        super().__init__(self.lagrange_grade, self.lagrange_coefs)
    
    def _get_lagrange_coefs(self):
        
        lagrange_poly = 0
        for i, yi in enumerate(self.y_values):
            wi = 1 
            xi = self.x_values[i]
            
            for j, xj in enumerate(self.x_values):
                if xi == xj:
                    continue
                numerator = Polinomios(n=1, coefs=[-xj, 1]) # x - xj
                denominator = xi - xj
                monomial = numerator // denominator
                wi *= monomial
                
            lagrange_poly += yi * wi
        
        return lagrange_poly.coefs
    
    
    def poly_plt(self, a, b, **kwargs):
        new_x_values = np.linspace(a, b, 100)
        new_y_values = [self(x) for x in new_x_values]
        plt.plot(new_x_values, new_y_values, **kwargs)
        plt.scatter(self.x_values, self.y_values)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.show()
        
    
        
      
        

#lag = Lagrange([(-1,0),(0,-1),(2,3),(-1,4)])
#print(lag)


#%%           

class Myarray():
    def __init__(self,lista,r,c, by_row = True):
        self.rows = r
        self.columns = c
        self.elems = lista
        self.orden = by_row
        if self.orden == False:
            self.orden = True
            self.elems = self.Orden().elems            
    
    def Orden(self): 
        new_elems = []
        for i in range(self.rows):
            row = self.elems[i::self.rows]
            for elem in row:
                new_elems.append(elem)
        return Myarray(new_elems, self.rows, self.columns, True)
    
    def __str__(self):
        matrix_str = ""
        for i in range(self.rows):
            for j in range(self.columns):
                matrix_str += str(self.GetElem(i+1, j+1)) + " "  
            matrix_str += "\n"
        return matrix_str
    
    def Identidad(self,j,k):
        if j != k:
            raise ValueError('Identity must be a square matrix')
        else:
            elems = [0]*(j*k)
            salida = Myarray(elems, j, k)        
            for i in range(len(elems)):
                coord = salida.GetCoords(i)
                if coord[0] == coord[1]:
                    elems[i] = 1                    
            return salida
    
    def __add__(self, b): 
        if isinstance(b, int) or isinstance(b,float):
            self.elems = [elem + b for elem in self.elems]
            salida = Myarray(self.elems, self.rows, self.columns)
        
        else:
            if not (self.rows == b.rows) and (self.columns == b.columns):
                raise ValueError('the number of rows must be equal to the number of columns')
            new_elems = []
            for i in range(len(self.elems)):
                new_elems.append(self.elems[i]+b.elems[i])
            salida = Myarray(new_elems, self.rows, self.columns)
        return salida
    
    def __radd__(self, b):
        return self.__add__(b)
    
    def __sub__(self,b): 
        if isinstance(b, int) or isinstance(b,float):
            self.elems = [elem - b for elem in self.elems]
            salida = Myarray(self.elems, self.rows, self.columns)
        else:
            if not (self.rows == b.rows) and (self.columns == b.columns):
                raise ValueError('the number of rows must be equal to the number of columns')
            new_elems = []
            for i in range(len(self.elems)):
                new_elems.append(self.elems[i]-b.elems[i])
            salida = Myarray(new_elems, self.rows, self.columns)        
        return salida
    
    def __rsub__(self, b):
        return self.__sub__(b)
    
    def __mul__(self, mult): #arreglar para cuando la otra matriz False
        new_elems = []
        if isinstance(mult, int) or isinstance(mult,float):
            for i in range(len(self.elems)):
                new_elems.append(self.elems[i]*mult)            
            salida = Myarray(new_elems, self.rows, self.columns)
            
        else:
            if self.columns != mult.rows:
                raise ValueError('the number of rows of array A must be equal to the number of columns in array B')
            for i in range(self.rows):
                fila = self.GetRow(i+1)
                for w in range(mult.columns):
                    col = mult.GetCol(w+1)
                    suma = 0
                    for x in range(len(fila)):
                        suma += (fila[x]*col[x])
                    new_elems.append(suma)                    
            salida = Myarray(new_elems, self.rows, mult.columns) 
        return salida
    
    def __rmul__(self, mult):
        return self.__mul__(mult)
    
    def __pow__(self,b):
        if isinstance(b, int) or isinstance(b,float):
            self.elems = [elem **b for elem in self.elems]
            salida = Myarray(self.elems, self.rows, self.columns)
        else:
            raise ValueError('the pow must be an int or float')    
        return salida
    
    def GetPos(self, j, k):
        check = j <= self.rows and k <= self.columns        
        if check == False:
            raise ValueError('rows or columns out of range')
        
        else:
            div = int(len(self.elems)/self.rows)
            index = div*(j-1)+k-1           
            return index 
        
       
    def GetCoords(self, m):
        check = m <= len(self.elems)-1               
        if check == False:
            raise ValueError('index out of range')
        
        else:
            ubicacion = m + 1 
            div = int(len(self.elems)/self.rows)                                
            cont = 0
            row = 1
            col = 1
            for i in range(1, len(self.elems)+1):       
                if ubicacion > i:
                    cont += 1
                    col += 1
                    if cont == div:
                        row += 1
                        col = 1
                        cont = 0
            salida = (row, col)
            
            return salida
                            
    def Switch(self):  
        new_elems = []
        if self.orden == True:
            for k in range(1,self.columns + 1):
                for j in range(1,self.rows + 1):
                    new_elems.append(self.elems[self.GetPos(j,k)])
        else:
            for j in range(1,self.rows + 1):
                for k in range(1,self.columns + 1):
                    new_elems.append(self.elems[self.GetPos(j,k)])  
                        
        return Myarray(self.elems, self.rows, self.columns, not (self.orden))
        
   
    def GetRow(self, j): 
        check = j <= self.rows              
        if check == False:
            raise ValueError('index out of range')
        
        else:
            div = int(len(self.elems)/self.rows)
            row = self.elems[div*(j-1):div*j]
            return row         
        
    def GetCol(self, k):
        check = k <= self.columns              
        if check == False:
            raise ValueError('index out of range')
        
        else:
            column = self.elems[k-1::self.columns]
            return column
            
        
    def GetElem(self,j,k):
        check = j <= self.rows and k <= self.columns        
        if check == False:
            raise ValueError('rows or columns out of range')
        
        else:
            indice = self.GetPos(j,k)
            elemento = self.elems[indice]                
            return elemento
            

    def GetSubmatrix(self, row_list, col_list):
        if not (isinstance(row_list, list) and isinstance(col_list, list)):
            raise ValueError('rowslist and colslist should be lists')
        
        new_mat = self
        cant_rows_eliminadas = 0
        for row in row_list:
            new_mat = new_mat.DelRow(row - cant_rows_eliminadas)
            cant_rows_eliminadas += 1
        
        cant_cols_eliminadas = 0
        for col in col_list:
            new_mat = new_mat.DelCol(col - cant_cols_eliminadas)
            cant_cols_eliminadas += 1
                    
        return new_mat
        

    def DelRow(self,j):
        check = self.rows >= j
        if check == False:
            raise ValueError ('index out of range')
        else:
            new_mat = []        
            for i in range(len(self.elems)):
                coord = self.GetCoords(i)
                x_coord = coord[0]
                if x_coord != j:
                    new_mat.append(self.elems[i])
        
            return Myarray(new_mat, self.rows-1, self.columns,self.orden)
    
    def DelRow2(self,j):
        check = self.columns >= j
        if check == False:
            raise ValueError ('index out of range')
        else:
            iden = self.Identidad(self.rows, self.columns)
            iden = iden.DelRow(j)            
            return iden*self
    
        
    def DelCol(self,k):
        check = self.columns >= k
        if check == False:
            raise ValueError ('index out of range')
        else:
            new_mat = []
            for i in range(len(self.elems)):
                coord = self.GetCoords(i)
                y_coord = coord[1]
                if y_coord != k:
                    new_mat.append(self.elems[i])
            
            return Myarray(new_mat,self.rows,self.columns-1,self.orden)
    
    def DelCol2(self,k):
        check = self.columns >= k
        if check == False:
            raise ValueError ('index out of range')
        else:
            iden = self.Identidad(self.rows, self.columns)
            iden = iden.DelCol(k)            
            return self*iden
               
    
    def SwapRows(self,j,k):
        check = self.rows >= j and self.columns >= k
        if check == False:
            raise ValueError('rows or columns out of range')
        else:
            iden =self.Identidad(self.rows,self.columns)
            
            pos_1 = iden.GetPos(j, j)
            pos_2 = iden.GetPos(k, k)
            pos_3 = iden.GetPos(j, k)
            pos_4 = iden.GetPos(k, j)
            
            iden.elems[pos_1] = 0
            iden.elems[pos_2] = 0
            iden.elems[pos_3] = 1
            iden.elems[pos_4] = 1
            
            return (iden*self)
    
    def SwapCols(self,j,k):
        check = self.rows >= j and self.columns >= k
        if check == False:
            raise ValueError('rows or columns out of range')
        
        else:
            iden =self.Identidad(self.rows,self.columns)
        
            pos_1 = iden.GetPos(j, j)
            pos_2 = iden.GetPos(k, k)
            pos_3 = iden.GetPos(j, k)
            pos_4 = iden.GetPos(k, j)
            
            iden.elems[pos_1] = 0
            iden.elems[pos_2] = 0
            iden.elems[pos_3] = 1
            iden.elems[pos_4] = 1
        
            return (self*iden)

    def ScaleRow(self,j,x):
        new_mat = []
        for i in range(len(self.elems)):
            x_coord = self.GetCoords(i)[0]
            if x_coord != j:
                new_mat.append(self.elems[i])
            else:
                new_mat.append(self.elems[i]*x)
        return Myarray(new_mat, self.rows, self.columns, self.orden)
        
    def ScaleCol(self,k,y):
        new_mat = []
        for i in range(len(self.elems)):
            y_coord = self.GetCoords(i)[1]
            if y_coord != k:
                new_mat.append(self.elems[i])
            else:
                new_mat.append(self.elems[i]*y)
        return Myarray(new_mat, self.rows, self.columns)
                
    def Traspose(self):
        salida = Myarray([0]*self.rows*self.columns, self.columns,self.rows,self.orden)
        for i in range(len(self.elems)):
            coords = salida.GetCoords(i)
            salida.elems[i] = self.GetElem(coords[1],coords[0])
        return salida 
       
    def FlipRows(self):
        new_elems = []
        for i in range(self.rows,0,-1):
            row = self.GetRow(i)
            new_elems.extend(row)
        return Myarray(new_elems, self.rows, self.columns)
    
    def FlipCols(self):
        new_elems = []
        for i in range(self.columns,0,-1):
            col = self.GetCol(i)
            new_elems.extend(col)
        return Myarray(new_elems, self.rows, self.columns)
            
    
    def Det(self):
        if not self.rows == self.columns:
            raise ValueError('the number of rows must be equal to the number of columns')
        else:
            if self.rows == 1:
                return self.elems[0]            
            sumatoria = 0 
            for col in range(1, self.columns + 1):
                f_arbitraria = 1
                submatriz = self.GetSubmatrix([f_arbitraria],[col])
                sumatoria += self.GetElem(f_arbitraria,col)*(-1)**((f_arbitraria)+col)* submatriz.Det()                
            return sumatoria
    
    def Inversa(self):
        if not self.rows == self.columns:
            raise ValueError('the number of rows must be equal to the number of columns')
        else:
            if self.Det() == 0:
                raise ValueError ('this array cannot be inversed, det = 0')
            else:
                lista_identidades = []
                
                for col in range(1, self.columns + 1):
                    identidad = self.Identidad(self.rows, self.columns)
                    columna = self.GetCol(col)
                    pivote = self.GetElem(col, col)                    
                    
                    if pivote == 0:
                        index = col
                        for elem in columna[col::]:
                            if elem != 0:
                                pivote = elem  
                                index += 1
                                break
                            index += 1
                            
                        self = self.SwapRows(col, index)
                        permutador = self.Identidad(self.columns, self.rows).SwapRows(col, index)
                        lista_identidades.append(permutador)                    
                        columna = self.GetCol(col)
                        pivote = self.GetElem(col, col) 
                        
                    for i in range(len(columna)): 
                        columna[i] = -columna[i]
                    columna[col-1] = 1
                    for i in range(len(columna)):
                        columna[i] = round(columna[i]/pivote, 2)
                    
                    contador = 0
                    for i in range(len(identidad.elems)):    
                        if identidad.GetCoords(i)[1] == col:                    
                            identidad.elems[i] = columna[contador]
                            contador += 1
                    
                    lista_identidades.append(identidad)
                    self = identidad*self  
                    
                inversa = 1
                for iden in lista_identidades[::-1]:
                    inversa = inversa*iden
                
                return inversa
                    
    
          
                    
        
elems1 = [1,4,1,0,0,3,1,1,6]
#elems2 = [2,8,7,6,5,3]
#elems3 = [2,8,7,6,5,3,9,5,6]

mat = Myarray(elems1, 3, 3, True)
#mat2 = Myarray(elems2, 3, 2, True)
#mat3 = Myarray(elems3, 3, 3, True)








class interpoly(Polinomios):
    def __init__(self, a, b):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            raise ValueError("a y b deben ser números reales.")
        if a >= b:
            raise ValueError("a debe ser menor que b.")
        self.a = a
        self.b = b
        


class linterp(interpoly):
    def __init__(self, puntos, a, b):
        self.puntos = sorted(puntos)
        super().__init__(a, b)
        
    def rectas(self):
        lista_rectas = []
        for i in range((len(self.puntos) - 1)):
            x0 = self.puntos[i][0]
            x1 = self.puntos[i+1][0]
            y0 = self.puntos[i][1]
            y1 = self.puntos[i+1][1]
            
            m = ((y1-y0)/(x1-x0))
            b = y0 + (((y1-y0)/(x1-x0))*(-x0))
            
            li = Polinomios(1, [m,b])
            
            lista_rectas.append(li)

        return lista_rectas

    def graficar(self, lista_rectas): 
        x_values = [[p[0]] for p in self.puntos]
        y_values = [[p[1]] for p in self.puntos]
        
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x_values, y_values, color = 'red', marker = 'o')
        
        plt.grid(True)
        plt.xlim(self.a, self.b)
        
        for i, recta in enumerate(lista_rectas):
            x0 = self.puntos[i][0]
            x1 = self.puntos[i+1][0]
            y0 = recta(x0)  
            y1 = recta(x1)  
            plt.plot([x0, x1], [y0, y1], label=f'Recta {i+1}')
            
        plt.legend([f'Recta {i+1}' for i in range(len(lista_rectas))])        
        plt.show()

puntos = [(1,2.718),(9,1.118),(5,1.221),(3,1.396),(7,1.154)]
poly  = linterp(puntos,0,9)

rectas = poly.rectas()
poly.graficar(rectas)

