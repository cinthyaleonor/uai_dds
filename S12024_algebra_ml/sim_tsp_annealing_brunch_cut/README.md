# 🚀 Simulador del Problema del Vendedor Viajero (TSP)

## 📝 Descripción

Este proyecto proporciona un marco de simulación y optimización integral para el **Problema del Vendedor Viajero (TSP)** utilizando dos técnicas algorítmicas, incluyendo el **Recocido Simulado** y un **métodos de Corte y Ramificación**:


- 🌡️ **Recocido Simulado (Simulated Annealing)**
- 🌳 **Branch and Cut** utilizando el solver **PuLP**

El simulador ofrece una comparación gráfica entre ambos métodos, permitiendo explorar diferentes estrategias de optimización.

## 🏗️ Estructura del Proyecto

### 📦 Clases Principales

1. **`TSPSintetico`**
   - 🎲 Genera instancias aleatorias del problema del vendedor viajero
   - 📏 Calcula la matriz de distancias entre ciudades

### 🧮 Algoritmos Implementados

1. **`recocido_simulado`**
   - 🔥 Resuelve el TSP mediante Recocido Simulado
   - ⚙️ Configurable con parámetros como:
     * Temperatura inicial
     * Factor de enfriamiento
     * Número de iteraciones
   - 📊 Genera historial de temperaturas y valores de función objetivo

2. **`branch_and_cut_iterativo`**
   - 🔬 Utiliza PuLP para resolver el TSP mediante programación lineal
   - 🏆 Encuentra la solución óptima global

3. **`comparar_esquemas_reduccion`**: 
  - Múltiples esquemas de reducción de temperatura (Exponencial, Logarítmico, Lineal)
  - Ajuste flexible de parámetros

4. **`comparar_rendimiento`**
   - 📈 Compara el progreso iterativo del Recocido Simulado
   - 🆚 Contrasta con la solución óptima de Branch and Cut
   - 🖼️ Genera visualizaciones de convergencia

## Estructura del repositorio

tsp_simulator_alods/
├── tsp_simulator.py       # Código fuente principal
├── README.md              # Documentación en Markdown
├── requirements.txt       # Dependencias del proyecto
└── .gitignore             # Archivos/carpetas a ignorar por Git


## 🛠️ Requisitos

Instala las dependencias necesarias:

```bash
pip install numpy pulp matplotlib
```

## ✨ Características

- **Optimización de Corte y Ramificación**: Método de resolución exacta utilizando Programación Lineal Entera
- **Comparación de Rendimiento**: Análisis exhaustivo de diferentes estrategias de solución
- **Herramientas de Visualización**: 
  - Evolución de la temperatura y la función objetivo
  - Gráficos comparativos de rendimiento

## 🛠 Prerrequisitos

- Python 3.8+
- Bibliotecas Requeridas:
  - NumPy
  - Matplotlib
  - PuLP
  - Pandas
  - Random
  - Time

## 🔍 Componentes Clave

### 1. Generación de Instancias TSP Sintéticas

La clase `TSPSintetico` genera instancias TSP aleatorias:
- Coordenadas aleatorias de ciudades en una cuadrícula de 100x100
- Cálculo de la distancia euclidiana
- Semilla flexible para reproducibilidad

### 2. Algoritmos de Enfriamiento Simulado

#### Esquemas de Reducción de Temperatura
- **Reducción Exponencial**: Enfriamiento gradual, geométrico
- **Reducción Logarítmica**: Enfriamiento más lento para más exploración
- **Reducción Lineal**: Disminución de temperatura consistente y predecible

### 3. Método de Corte y Ramificación

Una técnica de resolución exacta utilizando Programación Lineal Entera para encontrar la solución óptima.

## 🧪 Ejemplos de Uso

### Generar una Instancia TSP
```python
# Crear un problema TSP con 10 ciudades
problema = TSPSintetico(num_ciudades=10, semilla=42)
```

### Ejecutar Enfriamiento Simulado
```python
# Aplicar Enfriamiento Simulado con enfriamiento exponencial
mejor_ruta, mejor_distancia, temperaturas, valores = recocido_simulado(
    problema, 
    temperatura_inicial=7, 
    iteraciones_max=30
)
```

### Comparar Esquemas de Reducción de Temperatura
```python
# Visualizar y comparar diferentes estrategias de enfriamiento
comparar_esquemas_reduccion(problema)
```

### Comparación de Rendimiento
```python
# Comparar Enfriamiento Simulado y Corte y Ramificación
comparar_rendimiento(max_ciudades=15)
```

## 📦 Análisis de Rendimiento

El simulador proporciona métricas de rendimiento detalladas:
- Tiempos de ejecución
- Distancias de ruta
- Velocidad de convergencia
- Iteraciones de optimización

## 📄 Licencia

Distribuido bajo la Licencia MIT. Ver `LICENSE` para más información.

## 🎓 Referencias Académicas

- Aarts, Emile HL, Jan HM Korst, y Peter JM van Laarhoven. 1988. «A quantitative analysis of the simulated annealing algorithm: A case study for the traveling salesman problem». Journal of Statistical Physics 50: 187-206.
- Applegate, David L. 2006. The traveling salesman problem: a computational study. Vol. 17. Princeton university press.
- Bektas, Tolga. 2006. «The multiple traveling salesman problem: an overview of formulations and solution procedures». omega 34 (3): 209-19.
- Bertsimas, Dimitris, y Louis H Howell. 1993. «Further results on the probabilistic traveling salesman problem». European Journal of Operational Research 65 (1): 68-95.
- Bertsimas, Dimitris, y John N Tsitsiklis. 1997. Introduction to linear optimization. Vol. 6. Athena Scientific Belmont, MA.
- Dantzig, George, Ray Fulkerson, y Selmer Johnson. 1954. «Solution of a large-scale traveling-salesman problem». Journal of the operations research society of America 2 (4): 393-410.
- Gendreau, Michel, Jean-Yves Potvin, et al. 2010. Handbook of metaheuristics. Vol. 2. Springer.
- Kirkpatrick, Scott, C Daniel Gelatt Jr, y Mario P Vecchi. 1983. «Optimization by simulated annealing». science 220 (4598): 671-80.
- Laporte, Gilbert. 1992. «The traveling salesman problem: An overview of exact and approximate algorithms». European Journal of Operational Research 59 (2): 231-47.
- Menger, K. 1928. «Ein theorem über die bogenlänge». Anzeiger—Akademie der Wissenschaften in Wien—Mathematisch-naturwissenschaftliche Klasse 65: 264-66.
- Mitchell, Stuart, Michael OSullivan, y Iain Dunning. 2011. «Pulp: a linear programming toolkit for python». The University of Auckland, Auckland, New Zealand 65: 25.
- Pepper, Joshua W, Bruce L Golden, y Edward A Wasil. 2002. «Solving the traveling salesman problem with annealing-based heuristics: a computational study». IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans 32 (1): 72-77.
- Schrijver, Alexander. 2005. «On the History of Combinatorial Optimization (Till 1960)». En Discrete Optimization, editado por - K. Aardal, G. L. Nemhauser, y R. Weismantel, 12:1-68. Handbooks en Operations Research y Management Science. Elsevier. https://doi.org/https://doi.org/10.1016/S0927-0507(05)12001-5.
- Singh, Karanjot, Simran Kaur Bedi, y Prerna Gaur. 2020. «Identification of the most efficient algorithm to find Hamiltonian Path in practical conditions». En 2020 10th International Conference on Cloud Computing, Data Science & Engineering (Confluence), 38-44. https://doi.org/10.1109/Confluence47617.2020.9058283.
- The MathWorks, Inc. 2024. «What Is Simulated Annealing?» 2024. https://es.mathworks.com/help/gads/what-is-simulated-annealing.html.



## 🔗 Contacto

Cinthya Leonor Vergara Silva - [civergara@alumnos.uai.cl]

Enlace del Proyecto: [https://github.com/tuusuario/tsp-simulator](https://github.com/tuusuario/tsp-simulator)
