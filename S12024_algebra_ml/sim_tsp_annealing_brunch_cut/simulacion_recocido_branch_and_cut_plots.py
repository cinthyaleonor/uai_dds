import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import pulp as pl
import pandas as pd
from typing import List, Tuple, Callable

class TSPSintetico:
    """Clase para generar y gestionar instancias sintéticas del problema del vendedor viajero"""
    
    def __init__(self, num_ciudades: int, semilla: int = None):
        """
        Inicializa una instancia sintética del problema del vendedor viajero
        
        :param num_ciudades: Número de ciudades en el problema
        :param semilla: Semilla para reproducibilidad
        """
        if semilla is not None:
            np.random.seed(semilla)
        
        # Generar coordenadas aleatorias para las ciudades
        self.ciudades = np.random.rand(num_ciudades, 2) * 100
        
        # Calcular matriz de distancias
        self.matriz_distancias = self._calcular_matriz_distancias()
    
    def _calcular_matriz_distancias(self) -> np.ndarray:
        """
        Calcula la matriz de distancias euclídeas entre ciudades
        
        :return: Matriz de distancias
        """
        n = len(self.ciudades)
        matriz = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matriz[i, j] = np.linalg.norm(self.ciudades[i] - self.ciudades[j])
        
        return matriz
    
    def distancia_ruta(self, ruta: List[int]) -> float:
        """
        Calcula la distancia total de una ruta
        
        :param ruta: Lista de índices de ciudades
        :return: Distancia total de la ruta
        """
        distancia = 0
        for i in range(len(ruta) - 1):
            distancia += self.matriz_distancias[ruta[i], ruta[i+1]]
        
        # Añadir distancia de vuelta al punto de inicio
        distancia += self.matriz_distancias[ruta[-1], ruta[0]]
        
        return distancia

def esquema_reduccion_exponencial(temperatura: float, 
                                   iteracion: int, 
                                   alfa: float = 0.95) -> float:
    """
    Esquema de reducción de temperatura exponencial
    
    :param temperatura: Temperatura actual
    :param iteracion: Número de iteración
    :param alfa: Factor de reducción (entre 0 y 1)
    :return: Nueva temperatura
    """
    return temperatura * (alfa ** iteracion)

def esquema_reduccion_logaritmico(temperatura: float, 
                                   iteracion: int, 
                                   k: float = 1.0) -> float:
    """
    Esquema de reducción de temperatura logarítmico
    
    :param temperatura: Temperatura actual
    :param iteracion: Número de iteración
    :param k: Factor de escala
    :return: Nueva temperatura
    """
    return temperatura / (1 + k * math.log(1 + iteracion))

def esquema_reduccion_lineal(temperatura: float, 
                              iteracion: int, 
                              max_iteraciones: int = 30,
                              factor: float = 1.0) -> float:
    """
    Esquema de reducción de temperatura lineal
    
    :param temperatura: Temperatura inicial
    :param iteracion: Número de iteración
    :param max_iteraciones: Número máximo de iteraciones
    :param factor: Factor de reducción
    :return: Nueva temperatura
    """
    return max(0.01, temperatura * (1 - factor * iteracion / max_iteraciones))

def recocido_simulado(problema: TSPSintetico, 
                      temperatura_inicial: float = 7, 
                      iteraciones_max: int = 30,
                      esquema_reduccion: Callable = esquema_reduccion_exponencial) -> Tuple[List[int], float, List[float], List[float]]:
    """
    Implementación del algoritmo de Recocido Simulado para TSP.
    
    :param problema: Instancia del problema del vendedor viajero
    :param temperatura_inicial: Temperatura inicial
    :param iteraciones_max: Número máximo de iteraciones
    :param esquema_reduccion: Función de reducción de temperatura
    :return: Mejor ruta encontrada, su distancia, historial de temperaturas y valores de función objetivo
    """
    n = len(problema.ciudades)
    ruta_actual = list(range(n))
    random.shuffle(ruta_actual)
    
    mejor_ruta = ruta_actual.copy()
    mejor_distancia = problema.distancia_ruta(ruta_actual)
    
    temperatura = temperatura_inicial
    
    # Almacenar métricas para graficar
    temperaturas = [temperatura]
    valores_funcion_objetivo = [mejor_distancia]
    
    for iteracion in range(iteraciones_max):
        # Generar una solución vecina
        i, j = random.sample(range(n), 2)
        ruta_vecina = ruta_actual.copy()
        ruta_vecina[i], ruta_vecina[j] = ruta_vecina[j], ruta_vecina[i]
        
        distancia_vecina = problema.distancia_ruta(ruta_vecina)
        
        # Criterio de aceptación
        delta = distancia_vecina - problema.distancia_ruta(ruta_actual)
        
        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            ruta_actual = ruta_vecina
            
            if distancia_vecina < mejor_distancia:
                mejor_ruta = ruta_vecina.copy()
                mejor_distancia = distancia_vecina
        
        # Reducir temperatura
        temperatura = esquema_reduccion(temperatura_inicial, iteracion + 1)
        
        # Guardar métricas
        temperaturas.append(temperatura)
        valores_funcion_objetivo.append(problema.distancia_ruta(ruta_actual))
    
    return mejor_ruta, mejor_distancia, temperaturas, valores_funcion_objetivo

import matplotlib.pyplot as plt
import numpy as np

def comparar_esquemas_reduccion(problema: TSPSintetico, iteraciones_max=30, temperatura_inicial=7):
    """
    Compara diferentes esquemas de reducción de temperatura
    
    :param problema: Instancia del problema del vendedor viajero
    :param iteraciones_max: Número máximo de iteraciones
    :param temperatura_inicial: Temperatura inicial para el algoritmo de recocido simulado
    """
    # Parámetros base
    temperatura_inicial = temperatura_inicial
    iteraciones_max = iteraciones_max
    
    # Definir esquemas de reducción a comparar
    esquemas = [
        ("Exponencial", esquema_reduccion_exponencial),
        ("Logarítmico", esquema_reduccion_logaritmico),
        ("Lineal", esquema_reduccion_lineal)
    ]
    
    # Configuración de la visualización
    plt.figure(figsize=(10, 6))
    
    # Realizar simulaciones para cada esquema
    for idx, (nombre_esquema, esquema) in enumerate(esquemas, 1):
        # Ejecutar Recocido Simulado con el esquema actual
        _, mejor_distancia, temperaturas, valores_funcion_objetivo = recocido_simulado(
            problema, 
            temperatura_inicial=temperatura_inicial,
            iteraciones_max=iteraciones_max,
            esquema_reduccion=esquema
        )
        
        # Subplot para temperatura
        plt.subplot(3, 2, idx * 2 - 1)  # Poner en la columna 1
        plt.plot(temperaturas, label=f'{nombre_esquema} - Temperatura', color='blue')
        plt.title(f'Temperatura - {nombre_esquema}')
        plt.xlabel('Iteración')
        plt.ylabel('Temperatura')
        plt.legend()  # Agregar la leyenda
        
        # Subplot para función objetivo
        plt.subplot(3, 2, idx * 2)  # Poner en la columna 2
        plt.plot(valores_funcion_objetivo, label=f'{nombre_esquema} - Función Objetivo', color='orange')
        plt.title(f'Función Objetivo - {nombre_esquema}')
        plt.xlabel('Iteración')
        plt.ylabel('Distancia de Ruta')
        plt.legend()  # Agregar la leyenda
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de convergencia para cada esquema
    print("Análisis de Convergencia de Esquemas de Reducción de Temperatura:")
    for nombre_esquema, esquema in esquemas:
        # Ejecutar múltiples veces para obtener un promedio
        resultados = []
        for _ in range(10):
            _, mejor_distancia, _, valores_funcion_objetivo = recocido_simulado(
                problema, 
                temperatura_inicial=temperatura_inicial,
                iteraciones_max=iteraciones_max,
                esquema_reduccion=esquema
            )
            resultados.append(mejor_distancia)
        
        print(f"\n{nombre_esquema}:")
        print(f"Distancia promedio: {np.mean(resultados):.2f}")
        print(f"Desviación estándar: {np.std(resultados):.2f}")

## Branch and Cut

def branch_and_cut_iterativo(problema: TSPSintetico) -> Tuple[float, float]:
    """
    Implementación básica de Branch and Cut usando PuLP.
    Retorna la distancia óptima para el problema.
    
    :param problema: Instancia del problema del vendedor viajero
    :return: Distancia óptima y tiempo de solución
    """
    n = len(problema.ciudades)
    
    # Crear el modelo
    modelo = pl.LpProblem("Problema_Vendedor_Viajero", pl.LpMinimize)
    
    # Variables de decisión
    x = pl.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), cat='Binary')
    
    # Función objetivo
    modelo += pl.lpSum(x[i, j] * problema.matriz_distancias[i, j] 
                       for i in range(n) for j in range(n) if i != j)
    
    # Restricciones de entrada y salida
    for i in range(n):
        modelo += pl.lpSum(x[i, j] for j in range(n) if i != j) == 1
        modelo += pl.lpSum(x[j, i] for j in range(n) if i != j) == 1
    
    # Eliminación de subtours
    u = pl.LpVariable.dicts("u", range(n), lowBound=0, cat='Continuous')
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                modelo += u[i] - u[j] + n * x[i, j] <= n - 1
    
    # Resolver y medir tiempo
    inicio = time.time()
    modelo.solve()
    tiempo_total = time.time() - inicio
    
    # Reconstruir ruta (opcional)
    ruta = []
    actual = 0
    ruta.append(actual)
    
    while len(ruta) < n:
        for j in range(n):
            if j not in ruta and pl.value(x[actual, j]) > 0.5:
                ruta.append(j)
                actual = j
                break
    
    distancia_optima = problema.distancia_ruta(ruta)
    
    return distancia_optima, tiempo_total

def comparar_rendimiento(semilla_base: int = 42, max_ciudades: int = 15):
    """
    Compara el rendimiento de Recocido Simulado y Branch and Cut
    para diferentes tamaños de problemas.
    
    :param semilla_base: Semilla base para reproducibilidad
    :param max_ciudades: Número máximo de ciudades a probar
    """
    # Preparar estructuras para almacenar resultados
    resultados = []
    
    # Probar para cada número de ciudades
    for num_ciudades in range(5, max_ciudades + 1):
        # Crear problema con semilla reproducible
        problema = TSPSintetico(num_ciudades, semilla=semilla_base + num_ciudades)
        
        # Recocido Simulado
        inicio_sa = time.time()
        _, distancia_sa, temperaturas_sa, valores_sa = recocido_simulado(
            problema, 
            iteraciones_max=30, 
            temperatura_inicial=7
        )
        tiempo_sa = time.time() - inicio_sa
        
        # Encontrar en qué iteración se acerca más al óptimo
        iteracion_optimo_sa = None
        distancia_optima_bc, tiempo_bc = branch_and_cut_iterativo(problema)
        
        for i, valor in enumerate(valores_sa):
            if abs(valor - distancia_optima_bc) / distancia_optima_bc <= 0.05:  # Dentro del 5%
                iteracion_optimo_sa = i
                break
        
        # Almacenar resultados
        resultados.append({
            'Num_Ciudades': num_ciudades,
            'Distancia_SA': valores_sa[-1],
            'Distancia_BC': distancia_optima_bc,
            'Tiempo_SA': tiempo_sa,
            'Tiempo_BC': tiempo_bc,
            'Iteracion_Optimo_SA': iteracion_optimo_sa if iteracion_optimo_sa is not None else 30
        })
    
    # Convertir a DataFrame para facilitar análisis
    df_resultados = pd.DataFrame(resultados)
    
    # Graficar resultados
    plt.figure(figsize=(15, 10))
    
    # Tiempos de ejecución
    plt.subplot(2, 2, 1)
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Tiempo_SA'], marker='o', label='Recocido Simulado')
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Tiempo_BC'], marker='o', label='Branch and Cut')
    plt.xlabel('Número de Ciudades')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Tiempo de Ejecución')
    plt.legend()
    
    # Iteraciones para encontrar solución cercana al óptimo
    plt.subplot(2, 2, 2)
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Iteracion_Optimo_SA'], marker='o', color='green')
    plt.xlabel('Número de Ciudades')
    plt.ylabel('Iteraciones para Solución Cercana')
    plt.title('Iteraciones de Recocido Simulado')
    
    # Distancias
    plt.subplot(2, 2, 3)
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Distancia_SA'], marker='o', label='Recocido Simulado')
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Distancia_BC'], marker='o', label='Branch and Cut')
    plt.xlabel('Número de Ciudades')
    plt.ylabel('Distancia de la Ruta')
    plt.title('Distancia de la Ruta')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar tabla de resultados
    print(df_resultados)
    
    return df_resultados

