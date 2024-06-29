#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:35:40 2023

@author: abrilreggiardo
"""

def leer_file(file):
    """
    Esta funcion lee un archivo de entrada que en su primer linea tiene los nombres de 
    personas que luego llevaran a cabo durante todo el archivo movimientos de dinero, y las demás lineas
    por debajo tienen moviemintos anteriormente dichos. La función retorna dos variables, una sola con
    los nombres de las personas y otra una lista de movimeintos de dienro. 
    
    Parameters
    ----------
    file : archivo txt 

    Returns
    -------
    primera_linea : [str] con los nombres de las personas
    resto_archivo : [list] con todas las lineas de movimientos.

    """
    with open(file, 'r') as file:
        primera_linea = file.readline().strip().split(" ")
        resto_archivo = [line.strip("\n").split(" ") for line in file.readlines()]
        return (primera_linea, resto_archivo)
    
personas, datos_leidos = leer_file("transacciones_simple.txt")


datos = [
    [elemento for elemento in lista_interna if elemento] 
    for lista_interna in datos_leidos]


import matplotlib.pyplot as plt
import copy


dic = {}  # Diccionario con fechas y nombres
dic_personas = {}  # Nombres de propietarios

for i in personas:
    dic_personas[i] = 0

for lista in datos:
    fecha = lista[0]
    deudores = []

    if lista[1] == "*":
        nuevo_propietario = lista[2]
        personas.append(nuevo_propietario)
        dic_personas[nuevo_propietario] = 0

    elif lista[1].isalpha():
        acreedor = lista[1]
        monto = int(lista[2])

        if lista[3].isalpha():
            if len(lista) < 5:
                deudores.append(lista[3])
            else:
                deudores.extend(lista[3:])
        elif lista[3] == "~":
            if len(lista) < 5:
                deudores.extend(personas)
                deudores.remove(acreedor)
                print(deudores)
            else:
                deudores.extend(personas)
                excepcion_personas = lista[4:]
                print(deudores)
                deudores.remove(acreedor)
                for persona in excepcion_personas:
                    deudores.remove(persona)
           
        if monto > 0:
            cantidad_personas = len(deudores)
        else:
            continue
                
        cant_cd_prop = int(monto/cantidad_personas)
    
                # Restar el monto al acreedor y sumar la parte correspondiente a cada deudor
        dic_personas[acreedor] -= monto
        for deudor in deudores:
            dic_personas[deudor] += cant_cd_prop

    # Hacer una copia del estado acumulativo actual y asignarlo a dic[fecha]
    dic[fecha] = copy.deepcopy(dic_personas)
    print(f"Fecha: {fecha}, Saldos: {dic[fecha]}")


print(dic)

print("\nSaldos antes de crear el gráfico:")
for fecha, saldos in dic.items():
    print(f"Fecha: {fecha}, Saldos: {saldos}")


# Crear gráfico para cada persona
for persona in personas:
    if persona in dic[fecha]:
        lista_saldos = [dic[fecha][persona] for fecha in dic.keys()]
        plt.plot(list(dic.keys()), lista_saldos, label=persona)

plt.xlabel("Fecha")
plt.ylabel("Deuda en Pesos")
plt.title("Evolución de las Deudas")
plt.legend()
plt.show()