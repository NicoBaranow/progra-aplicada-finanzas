import math
import time
#2) Fibonacci
# a) Construya una función recursiva Fib1(n) que compute la secuencia de Fibonacci.

# b) Construya una función Fib2(n) que compute la secuencia de Fibonacci usando la solución analítica, y construya una tabla de
# F(n) para 1 <= n <= 20 con ambos métodos. Compare los resultados obtenidos.

# c) Y ad·ptelo para comparar el tiempo que tarda la funcion recursiva en computar los primeros
# 40 tÈrminos de la secuencia de Fibonacci, versus lo que tarda la fÛrmula explÌcita.

def Fib1(n):
    start = time.perf_counter_ns()
    if n <= 0:
        end = time.perf_counter_ns()
        return 0
    elif n == 1:
        end = time.perf_counter_ns()
        return 1
    else:
        end = time.perf_counter_ns()
        fibo = (Fib1(n-1) + Fib1(n-2))
        return fibo, start, end
    
def Fib2(n):
    start = time.perf_counter_ns()
    fibo = int(1/math.sqrt(5) * (((1+math.sqrt(5))/2)**n - ((1-math.sqrt(5))/2)**n))
    end = time.perf_counter_ns()
    return fibo, start, end

def secuencia(n):
    fibonacci = []
    for i in range(n):
        fibonacci.append(Fib1(i))
    return fibonacci

def tablaComparativa(n):
    print("Fib1     vs      Fib2")
    for i in range(n):
        fibo1,start1,end1 = Fib1(i)
        fibo2,start2,end2 = Fib2(i)
        
        print("Fibonacci de", i)
        print("Fib1 para Fibonacci de", i, "tarda:" ,end1-start1 , "Y el resultado es", fibo1,"\n\n")
        print("Fib2 para Fibonacci de", i, "tarda:" ,end2-start2 , "Y el resultado es", fibo2)

tablaComparativa(20)
    
