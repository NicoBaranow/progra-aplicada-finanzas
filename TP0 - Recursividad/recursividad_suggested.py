# ej 1 Hanoi

def Hanoi(n,i,j):
    """ n es el numero de discos, i vara de salida, j vara de llegada
        implicitamente la vara ausente es la del medio """
    if n==1:
        lista.append((n,i,j))        
    else:
        k= list(set([1,2,3])-set([i,j]))[0]
        Hanoi(n-1,i,k)
        lista.append((n,i,j))
        Hanoi(n-1,k,j)
    return None
        
        

#%% ej 2 Fibonacci

def fib1(n):
    if n == 0:
        salida = 0 
        
    elif n == 1:
        salida = 1
    
    else:
        salida = fib1(n-1) + fib1(n-2)
    
    return salida
        

#print(fib1(4))


#%% Maximo comun divisor

def mcd(a,b):
    if a == b:
        salida = a
    
    else:
        menor = min(a,b)
        mayor = max(a,b)
        if mayor % menor == 0:
            salida = menor
        else:
            salida = mcd(menor,mayor % menor)
               
    return salida

#print(mcd(8,7))



#%% ej 4 a


def r2i(romano):
    numero = 0
    dic_valores = {'m':1000,'d':500, 'c':100,'l':50,'x':10,'v':5,'i':1 }
    
    if len(romano) == 1:
        for posible in dic_valores:
            if posible == romano[0]:
                numero = dic_valores[posible]
    else:
        lista_romano = list(romano)
        
        letra_actual = lista_romano[0]
        letra_siguiente = lista_romano[1]
        
        if dic_valores[letra_actual] > dic_valores[letra_siguiente]:
            numero += dic_valores[letra_actual] + r2i(romano[1::])
        
        elif dic_valores[letra_actual] == dic_valores[letra_siguiente]: 
            numero += dic_valores[letra_actual] + r2i(romano[1::])
            
        else:
            numero += r2i(romano[1::]) - dic_valores[letra_actual] 
                				
    return numero
  
#print(r2i('mcdxiiii'))

def r2i2(roman):

    roman_to_int = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
    
    # Caso base: cadena vacía
    if not roman:
        return 0
    
    # Si la cadena tiene un solo carácter, devuelve su valor
    if len(roman) == 1:
        return roman_to_int[roman]
    
    # Caso de sustracción: numeral menor antes de uno mayor
    if roman_to_int[roman[0]] < roman_to_int[roman[1]]:
        return r2i(roman[1:]) - roman_to_int[roman[0]]
    
    # Caso de adición: numeral mayor o igual antes
    return roman_to_int[roman[0]] + r2i(roman[1:])

#%% ej 4 b

def i2r(numero):

    roman_numerals = [('M', 1000), ('D', 500), ('C', 100), ('L', 50), ('X', 10), ('V', 5), ('I', 1)]
    
    substractive_numerals = [('C', 100), ('X', 10), ('I', 1)]

    if numero == 0:
        return ""
    
    # Encuentra el mayor valor romano menor o igual que 'numero'
    for numeral, value in roman_numerals:
        if value <= numero:
            return numeral + i2r(numero - value)
        
        # Verifica si puede ser escrito como sustracción
        for substraction_numeral, substraction_value in substractive_numerals:
            # Evita la repetición de pares de sustracción
            if value * 0.1 <= substraction_value < value:
                substracted_value = value - substraction_value
                # Asegura un numeral de sustracción válido
                if 0 < substracted_value <= numero:
                    return substraction_numeral + numeral + i2r(numero - substracted_value)

#%% elementos quimicos

elementos_quimicos = ['Actinium', 'Aluminum', 'Americium', 'Antimony', 'Argon', 'Arsenic'
     , 'Astatine', 'Barium', 'Berkelium', 'Beryllium', 'Bismuth', 'Bohrium'
     , 'Boron', 'Bromine', 'Cadmium', 'Calcium', 'Californium', 'Carbon'
     , 'Cerium', 'Cesium', 'Chlorine', 'Chromium', 'Cobalt', 'Copernicium'
     , 'Copper', 'Curium', 'Darmstadtium', 'Dubnium', 'Dysprosium', 'Einsteinium'
     , 'Erbium', 'Europium', 'Fermium', 'Flerovium', 'Fluorine', 'Francium'
     , 'Gadolinium', 'Gallium', 'Germanium', 'Gold', 'Hafnium', 'Hassium'
     , 'Helium', 'Holmium', 'Hydrogen', 'Indium', 'Iodine', 'Iridium', 'Iron'
     , 'Krypton', 'Lanthanum', 'Lawrencium', 'Lead', 'Lithium', 'Livermorium'
     , 'Lutetium', 'Magnesium', 'Manganese', 'Meitnerium', 'Mendelevium', 'Mercury'
     , 'Molybdenum', 'Moscovium', 'Neodymium', 'Neon', 'Neptunium', 'Nickel'
     , 'Nihonium', 'Niobium', 'Nitrogen', 'Nobelium', 'Oganesson', 'Osmium'
     , 'Oxygen', 'Palladium', 'Phosphorus', 'Platinum', 'Plutonium', 'Polonium'
     , 'Potassium', 'Praseodymium', 'Promethium', 'Protactinium', 'Radium', 'Radon'
     , 'Rhenium', 'Rhodium', 'Roentgenium', 'Rubidium', 'Ruthenium', 'Rutherfordium'
     , 'Samarium', 'Scandium', 'Seaborgium', 'Selenium', 'Silicon', 'Silver'
     , 'Sodium', 'Strontium', 'Sulfur', 'Tantalum', 'Technetium', 'Tellurium'
     , 'Tennessine', 'Terbium', 'Thallium', 'Thorium', 'Thulium', 'Tin', 'Titanium'
     , 'Tungsten', 'Uranium', 'Vanadium', 'Xenon', 'Ytterbium', 'Yttrium', 'Zinc'
     , 'Zirconium']

def encontrar_secuencia_mas_larga(elemento, elementos_quimicos_set, secuencia=[]):

    secuencia.append(elemento)

    # Buscar posibles candidatos para continuar la secuencia
    secuencia_set = set(secuencia)
    candidatos_posibles = elementos_quimicos_set - secuencia_set
    candidatos = [e for e in candidatos_posibles if e.startswith(elemento[-1].upper())]

    # Variable para guardar la secuencia más larga encontrada
    secuencia_mas_larga = list(secuencia)

    # Explorar cada candidato para continuar la secuencia
    for candidato in candidatos:
        nueva_secuencia = encontrar_secuencia_mas_larga(candidato, elementos_quimicos_set, list(secuencia))
        if len(nueva_secuencia) > len(secuencia_mas_larga):
            secuencia_mas_larga = nueva_secuencia

    # Devolver la secuencia más larga encontrada
    return secuencia_mas_larga

test_nombre_elemento = 'Iron'
test_secuencia = encontrar_secuencia_mas_larga(test_nombre_elemento, set(elementos_quimicos), secuencia=[])

#%% suma de digitos de un numero 

def sum_of_digits(num):
    if num == 0:
        return 0
    else:
        return (num % 10) + sum_of_digits(num//10)


#%% calculating the power of a number 

def power(num, p):
    if p == 0:
        return 1
    elif p == 1: 
        return num
    else:
        return num * power(num, p-1)

#%% calculating the square root of a number

def rec_sqrt(x,n):
    if n == 0:
        return 0 
    else:
        return (1/n) + rec_sqrt(x - (1/n), n + 1)
    
#%% calculating the sum of elements in a list 

def sum_list(lst):
    if not lst:
        return 0
    else:
        return lst[0] + sum_list(lst[1:])
    
#%% calculating the legth of a list

def length(lst):
    if not lst:
        return 0
    else:
        return 1 + length(lst[1:])

#%% calcular la suma de los n numeros naturales anteriores

def rec_sum_natural(n):
    if n == 0: 
        return 0 
    else:
        return n + rec_sum_natural(n-1)

#%% calcular la suma de los primeros n numeros impares 

def rec_sum_odd(n):
    if n == 0:
        return 0
    else:
        return (2 * n - 1) + rec_sum_odd(n - 1)

#%% calcular la suma de los primeros n numeros pares

def rec_sum_even(n):
    if n == 0:
        return 0
    else:
        return (2 * n) + rec_sum_odd(n - 1)

#%% calcular la sumade los primeros n terminos de una secuencia de fibonacci

def rec_sem_fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1 
    else:
        return rec_sem_fib(n-1)+ rec_sem_fib(n-2)

#%% calcular el maximo de dos numeros

def rec_max(x,y):
    if x > y:
        return x
    else:
        return rec_max(x + 1, y)

#%% calcular el minimo de dos numeros

def rec_min(x,y):
    if x < y:
        return x
    else:
        return rec_min(x - 1, y)
    



#%% ej 7

def raiz(n, guess=1.0):
    resultado = 'guess'
    if abs(n - guess**2)<= 10**(-12):
        return resultado
    else:
        return raiz(n, ((guess + (n/guess))/2))

#var1 = raiz(3)
#print(var1)

#%% 

def resolver_rompecabezas_caballo(n, m, x, y, path=None, visited=None):
    # Si el tablero no tiene tamaño o la posición es inválida, retorna False
    if not n or not m or x < 0 or y < 0 or x >= n or y >= m:
        return False

    # Si ya se ha visitado todas las casillas, retorna True
    if len(path) == n * m:
        return True

    # Si la casilla actual ya ha sido visitada, retorna False
    if visited and (x, y) in visited:
        return False

    # Define el movimiento del caballo en forma de "L"
    movimientos = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

    # Ordena los movimientos en sentido horario
    movimientos = sorted(movimientos, key=lambda x: (x[0], x[1]))

    # Marca la casilla actual como visitada
    if not visited:
        visited = set()
    visited.add((x, y))

    # Agrega la casilla actual al camino
    path.append((x, y))

    # Prueba cada movimiento posible
    for movimiento in movimientos:
        # Si el movimiento lleva a una casilla fuera del tablero, sigue con el siguiente movimiento
        if not (0 <= x + movimiento[0] < n and 0 <= y + movimiento[1] < m):
            continue

        # Si el movimiento lleva a una casilla ya visitada, sigue con el siguiente movimiento
        if visited and (x + movimiento[0], y + movimiento[1]) in visited:
            continue

        # Si el movimiento lleva a una casilla no visitada, intenta resolver el rompecabezas recursivamente
        if resolver_rompecabezas_caballo(n, m, x + movimiento[0], y + movimiento[1], path, visited):
            return True

    # Si no se pudo resolver el rompecabezas con ningún movimiento, elimina la casilla actual del camino
    # y retorna False
    path.pop()
    return False

#%% 

def generar_permutaciones(lista):
    # Si la lista tiene menos de 2 elementos, retorna la lista misma
    if len(lista) < 2:
        return [lista]

    # Crea una lista vacía para almacenar las permutaciones
    permutaciones = []

    # Iterar sobre cada elemento de la lista
    for i, elemento in enumerate(lista):
        # Obtener el resto de la lista sin el elemento actual
        resto = lista[:i] + lista[i+1:]

        # Generar las permutaciones recursivamente con el resto de la lista
        permutaciones_resto = generar_permutaciones(resto)

        # Agregar el elemento actual al inicio de cada permutación del resto de la lista
        for permutacion in permutaciones_resto:
            permutaciones.append([elemento] + permutacion)

    # Retornar la lista de permutaciones
    return permutaciones

#%% 

def encontrar_combinaciones(numeros, target, combinacion=[], combinaciones=None):
    # Si la combinación actual alcanza el objetivo, agregarlo a la lista de combinaciones
    if target == 0:
        if combinaciones is not None:
            combinaciones.append(combinacion)
        else:
            return [combinacion]

    # Si el objetivo es negativo o la lista de números está vacía, retornar una lista vacía
    if target < 0 or not numeros:
        return []

    # Si la lista de combinaciones no se proporcionó como argumento, crear una nueva lista
    if combinaciones is None:
        combinaciones = []

    # Iterar sobre cada número en la lista
    for i, numero in enumerate(numeros):
        # Crear una nueva combinación sin el número actual
        nueva_combinacion = combinacion.copy()
        nueva_combinacion.append(numero)

        # Encontrar las combinaciones recursivamente con el resto de la lista y el objetivo actual menos el número actual
        encontrar_combinaciones(numeros[i+1:], target - numero, nueva_combinacion, combinaciones)

    # Retornar la lista de combinaciones
    return combinaciones

#%% 

def encontrar_particiones(lista, particion=[], particiones=None):
    # Si la lista está vacía, agregar la partición actual a la lista de particiones
    if not lista:
        if particiones is not None:
            particiones.append(particion)
        else:
            return [particion]

    # Si la lista de particiones no se proporcionó como argumento, crear una nueva lista
    if particiones is None:
        particiones = []

    # Iterar sobre cada elemento en la lista
    for i, elemento in enumerate(lista):
        # Crear una nueva partición sin el elemento actual
        nueva_particion = particion.copy()
        nueva_particion.append(elemento)

        # Encontrar las particiones recursivamente con el resto de la lista
        encontrar_particiones(lista[:i] + lista[i+1:], nueva_particion, particiones)

    # Retornar la lista de particiones
    return particiones

#%% A nested list is a list that contains other lists. 
#   Flattening a nested list means to convert it into a one-dimensional list. 

def flatten_list(nested_list):
    
    flat_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flat_list.extend(flatten_list(sublist))
        else:
            flat_list.append(sublist)
    return flat_list

#%% calcular la len de una lista de forma recursiva

def length_list(lst):
    
    if not lst:
        return 0
    else:
        return 1 + length_list(lst[1:])

#%% calcular el maximo elemento de una lista

def find_max(my_list):
    if len(my_list) == 0:
        return None
    
    if len(my_list) == 1:
        return my_list[0]
    temp = my_list[0]
    
    if my_list[0] > my_list[1]:
        my_list[1] = temp
        
    return find_max(my_list[1:])
    


#%% calcular el minimo elemento de una lista

def find_min(my_list):
    if len(my_list) == 0:
        return None
    
    if len(my_list) == 1:
        return my_list[0]
    temp = my_list[0]
    
    if my_list[0] < my_list[1]:
        my_list[1] = temp
        
    return find_min(my_list[1:])

#%%

def recursive_for(func, lst):
    if not lst:
        return
    func(lst[0])
    recursive_for(func, lst[1:])
    
#recursive_for(print, [1,2,3,4,5,6])

#%% 

def ordenar(lista):
    if len(lista) == 2:
        if lista[0]<lista[1]:
            salida = lista
        else:
            salida =[lista[1],lista[0]]
    else:
       l0 = lista[0]
       aux = ordenar(lista[1:len(lista)])
       
       if l0 <= aux[0]:
           salida = [l0]+aux
       elif l0>= aux[len(aux)-1]:
           salida  = aux + [l0]
       else:
           salida = [aux[0]]+ordenar([lista[0]]+aux[1:len(aux)])
       #print(salida)
    return salida
            
lista = [(0,1),(1,2),(0,2),(0,3),(1,1),(1,0),(1,1)]
print(lista)
salida = ordenar(lista)
print(salida)

#%% 

def es_palindromo(cadena):
   
    if len(cadena) <= 1:
        resultado = True
    else:
        primer_letra = cadena[0]
        ultima_letra = cadena[-1]
        if primer_letra == ultima_letra:
            letras_intermedias = cadena[1:-1]
            resultado = es_palindromo(letras_intermedias)
        else:
            resultado = False
    return resultado



