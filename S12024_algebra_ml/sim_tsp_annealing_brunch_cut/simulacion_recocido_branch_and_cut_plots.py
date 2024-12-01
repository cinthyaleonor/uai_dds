import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import pulp as pl
from typing import List, Tuple, Callable
import pandas as pd


class TSPSintetico:
    """Clase para generar y gestionar instancias sintéticas del problema del vendedor viajero"""
    
    def __init__(self, num_ciudades: int, semilla: int = None):
        if semilla is not None:
            np.random.seed(semilla)
        self.ciudades = np.random.rand(num_ciudades, 2) * 100
        self.matriz_distancias = self._calcular_matriz_distancias()
    
    def _calcular_matriz_distancias(self) -> np.ndarray:
        n = len(self.ciudades)
        matriz = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matriz[i, j] = np.linalg.norm(self.ciudades[i] - self.ciudades[j])
        return matriz
    
    def distancia_ruta(self, ruta: List[int]) -> float:
        distancia = 0
        for i in range(len(ruta) - 1):
            distancia += self.matriz_distancias[ruta[i], ruta[i + 1]]
        distancia += self.matriz_distancias[ruta[-1], ruta[0]]
        return distancia


def esquema_reduccion_exponencial(temperatura: float, iteracion: int, alfa: float = 0.95) -> float:
    return temperatura * (alfa ** iteracion)


def esquema_reduccion_logaritmico(temperatura: float, iteracion: int, k: float = 1.0) -> float:
    return temperatura / (1 + k * math.log(1 + iteracion))


def esquema_reduccion_lineal(temperatura: float, iteracion: int, max_iteraciones: int = 30) -> float:
    return max(0.01, temperatura * (1 - iteracion / max_iteraciones))


def recocido_simulado(problema: TSPSintetico, 
                      temperatura_inicial: float = 7, 
                      iteraciones_max: int = 30, 
                      esquema_reduccion: Callable = esquema_reduccion_exponencial) -> Tuple[List[int], float, List[float], List[float]]:
    n = len(problema.ciudades)
    ruta_actual = list(range(n))
    random.shuffle(ruta_actual)

    mejor_ruta = ruta_actual.copy()
    mejor_distancia = problema.distancia_ruta(ruta_actual)

    temperatura = temperatura_inicial
    temperaturas = [temperatura]
    valores_funcion_objetivo = [mejor_distancia]

    for iteracion in range(iteraciones_max):
        i, j = random.sample(range(n), 2)
        ruta_vecina = ruta_actual.copy()
        ruta_vecina[i], ruta_vecina[j] = ruta_vecina[j], ruta_vecina[i]

        distancia_vecina = problema.distancia_ruta(ruta_vecina)
        delta = distancia_vecina - problema.distancia_ruta(ruta_actual)

        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            ruta_actual = ruta_vecina
            if distancia_vecina < mejor_distancia:
                mejor_ruta = ruta_vecina.copy()
                mejor_distancia = distancia_vecina

        temperatura = esquema_reduccion(temperatura_inicial, iteracion + 1)
        temperaturas.append(temperatura)
        valores_funcion_objetivo.append(problema.distancia_ruta(ruta_actual))
    
    return mejor_ruta, mejor_distancia, temperaturas, valores_funcion_objetivo

def mostrar_resultados_esquemas(problema: TSPSintetico, iteraciones_max=30, temperatura_inicial=7):
    esquemas = [
        ("Exponencial", esquema_reduccion_exponencial),
        ("Logarítmico", esquema_reduccion_logaritmico),
        ("Lineal", esquema_reduccion_lineal)
    ]
    
    for nombre_esquema, esquema in esquemas:
        # Ejecutar Recocido Simulado con el esquema de reducción
        _, mejor_distancia, temperaturas, valores_funcion_objetivo = recocido_simulado(
            problema,
            temperatura_inicial=temperatura_inicial,
            iteraciones_max=iteraciones_max,
            esquema_reduccion=esquema
        )
        
        # Mostrar resultados de cada esquema
        print(f"\nResultados para el esquema de reducción: {nombre_esquema}")
        print(f"{'Iteración':<10} {'Temperatura':<15} {'Valor Función Objetivo'}")
        print("="*50)
        
        for i in range(len(temperaturas)):
            print(f"{i:<10} {temperaturas[i]:<15.2f} {valores_funcion_objetivo[i]:<20.2f}")
        
        print(f"\nMejor distancia encontrada: {mejor_distancia:.2f}")
        print("-"*50)


def comparar_esquemas_reduccion(problema: TSPSintetico, iteraciones_max=30, temperatura_inicial=7):
    esquemas = [
        ("Exponencial", esquema_reduccion_exponencial),
        ("Logarítmico", esquema_reduccion_logaritmico),
        ("Lineal", esquema_reduccion_lineal)
    ]
    plt.figure(figsize=(12, 8))
    for idx, (nombre_esquema, esquema) in enumerate(esquemas):
        _, mejor_distancia, temperaturas, valores_funcion_objetivo = recocido_simulado(
            problema,
            temperatura_inicial=temperatura_inicial,
            iteraciones_max=iteraciones_max,
            esquema_reduccion=esquema
        )
        plt.subplot(3, 2, idx * 2 + 1)
        plt.plot(temperaturas, label=f'Temperatura - {nombre_esquema}', color='blue')
        plt.xlabel('Iteración')
        plt.ylabel('Temperatura')
        plt.legend()
        plt.subplot(3, 2, idx * 2 + 2)
        plt.plot(valores_funcion_objetivo, label=f'Función Objetivo - {nombre_esquema}', color='orange')
        plt.xlabel('Iteración')
        plt.ylabel('Distancia')
        plt.legend()
    plt.tight_layout()
    plt.show()


def branch_and_cut_iterativo(problema: TSPSintetico) -> Tuple[float, float]:
    n = len(problema.ciudades)
    modelo = pl.LpProblem("Problema_Vendedor_Viajero", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), cat='Binary')
    modelo += pl.lpSum(x[i, j] * problema.matriz_distancias[i, j] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        modelo += pl.lpSum(x[i, j] for j in range(n) if i != j) == 1
        modelo += pl.lpSum(x[j, i] for j in range(n) if i != j) == 1

    u = pl.LpVariable.dicts("u", range(n), lowBound=0, cat='Continuous')
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                modelo += u[i] - u[j] + n * x[i, j] <= n - 1

    inicio = time.time()
    modelo.solve()
    tiempo_total = time.time() - inicio
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
    para diferentes tamaños de problemas TSP.

    Args:
        semilla_base (int): Semilla base para reproducibilidad.
        max_ciudades (int): Número máximo de ciudades a probar.

    Returns:
        pd.DataFrame: DataFrame con los resultados comparativos.
    """
    # Preparar estructura para almacenar resultados
    resultados = []

    # Probar para diferentes tamaños de problemas (de 5 a max_ciudades)
    for num_ciudades in range(5, max_ciudades + 1):
        # Crear problema TSP sintético
        problema = TSPSintetico(num_ciudades, semilla=semilla_base + num_ciudades)

        # Resolver usando Recocido Simulado
        inicio_sa = time.time()
        _, distancia_sa, temperaturas_sa, valores_sa = recocido_simulado(
            problema, 
            iteraciones_max=30, 
            temperatura_inicial=7
        )
        tiempo_sa = time.time() - inicio_sa

        # Resolver usando Branch and Cut
        inicio_bc = time.time()
        distancia_optima_bc, tiempo_bc = branch_and_cut_iterativo(problema)  # Función adicional requerida
        tiempo_bc = time.time() - inicio_bc

        # Encontrar en qué iteración del Recocido Simulado se aproxima al óptimo
        iteracion_optimo_sa = None
        for i, valor in enumerate(valores_sa):
            if abs(valor - distancia_optima_bc) / distancia_optima_bc <= 0.05:  # Tolerancia del 5%
                iteracion_optimo_sa = i
                break

        # Almacenar los resultados
        resultados.append({
            'Num_Ciudades': num_ciudades,
            'Distancia_SA': valores_sa[-1],
            'Distancia_BC': distancia_optima_bc,
            'Tiempo_SA': tiempo_sa,
            'Tiempo_BC': tiempo_bc,
            'Iteracion_Optimo_SA': iteracion_optimo_sa if iteracion_optimo_sa is not None else 30
        })

    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Graficar los resultados
    plt.figure(figsize=(15, 10))

    # Tiempo de ejecución
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

    # Distancia de la ruta
    plt.subplot(2, 2, 3)
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Distancia_SA'], marker='o', label='Recocido Simulado')
    plt.plot(df_resultados['Num_Ciudades'], df_resultados['Distancia_BC'], marker='o', label='Branch and Cut')
    plt.xlabel('Número de Ciudades')
    plt.ylabel('Distancia de la Ruta')
    plt.title('Distancia de la Ruta')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Mostrar los resultados en forma de tabla
    print(df_resultados)

    return df_resultados


if __name__ == "__main__":

    # Configuración para mostrar todas las columnas y filas
    pd.set_option('display.max_columns', None)  # Muestra todas las columnas
    pd.set_option('display.max_rows', None)     # Muestra todas las filas
    pd.set_option('display.width', 1000)        # Ajusta el ancho de la tabla en la consola

    problema = TSPSintetico(num_ciudades=10, semilla=42)

    # Ejemplo de uso de la función
    problema = TSPSintetico(num_ciudades=10, semilla=42)
    mostrar_resultados_esquemas(problema)

    comparar_esquemas_reduccion(problema)
    optimo_bc, tiempo_bc = branch_and_cut_iterativo(problema)
    print(f"Branch and Cut - Distancia óptima: {optimo_bc}, Tiempo: {tiempo_bc:.2f} segundos")

    comparar_rendimiento(42, 15)
