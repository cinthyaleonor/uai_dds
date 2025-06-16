# Resolución de Ejercicios de Jackknife

## Ejercicio 1: Sesgo del Estimador Jackknife

**Problema:** Se tiene una muestra aleatoria $X_1, \ldots, X_n$ asociada a cierta población de parámetro $\mu$. Sea $\hat{\theta} = \bar{x}$ una estimación de $\mu$. Mostrar que al realizar la estimación del parámetro $\mu$ empleando el método de Jackknife, el sesgo asociado al estimador $\hat{\theta}_{Jack}$ es 0.

**Solución:**

El estimador Jackknife se define como:

$$
\hat{\theta}_{Jack} = n\hat{\theta} - (n-1)\hat{\theta}_{(-i)}
$$

donde $\hat{\theta}_{(-i)}$ es el estimador calculado omitiendo la observación $i$-ésima.

Para la media muestral:

- $\hat{\theta} = \bar{x} = \frac{1}{n}\sum_{i=1}^n X_i$
- $\hat{\theta}_{(-i)} = \bar{x}_{(-i)} = \frac{1}{n-1}\sum_{j \neq i} X_j$

Calculamos $\hat{\theta}_{(-i)}$:

$$
\hat{\theta}_{(-i)} = \frac{1}{n-1}\left(\sum_{j=1}^n X_j - X_i\right) = \frac{n\bar{x} - X_i}{n-1}
$$

El estimador Jackknife promedio es:

$$\hat{\theta}_{Jack} = \frac{1}{n}\sum_{i=1}^n \left[n\hat{\theta} - (n-1)\hat{\theta}_{(-i)}\right]$$

Sustituyendo:

$$\hat{\theta}_{Jack} = \frac{1}{n}\sum_{i=1}^n \left[n\bar{x} - (n-1)\frac{n\bar{x} - X_i}{n-1}\right]$$

$$= \frac{1}{n}\sum_{i=1}^n \left[n\bar{x} - (n\bar{x} - X_i)\right]$$

$$= \frac{1}{n}\sum_{i=1}^n X_i = \bar{x}$$

Por lo tanto: $E[\hat{\theta}_{Jack}] = E[\bar{x}] = \mu$

**Conclusión:** El sesgo es $Sesgo(\hat{\theta}_{Jack}) = E[\hat{\theta}_{Jack}] - \mu = 0$

---

## Ejercicio 2: Distribución Discreta

**Parámetros:** $r > 0$ conocido, $\theta \in [0,1]$ a estimar

**Función de probabilidad:**
$$p_X(x;\theta) = \binom{r+x-1}{x}(1-\theta)^x \theta^r \quad ; \quad x = 0,1,2,3,\ldots$$

### a) Estimador Máximo Verosímil

La función de verosimilitud para una muestra de tamaño $n$:
$$L(\theta) = \prod_{i=1}^n \binom{r+x_i-1}{x_i}(1-\theta)^{x_i} \theta^r$$

$$= \theta^{nr}(1-\theta)^{\sum_{i=1}^n x_i} \prod_{i=1}^n \binom{r+x_i-1}{x_i}$$

La log-verosimilitud:
$$\ell(\theta) = nr\ln(\theta) + \left(\sum_{i=1}^n x_i\right)\ln(1-\theta) + C$$

Derivando e igualando a cero:
$$\frac{d\ell}{d\theta} = \frac{nr}{\theta} - \frac{\sum_{i=1}^n x_i}{1-\theta} = 0$$

Resolviendo:
$$\frac{nr}{\theta} = \frac{\sum_{i=1}^n x_i}{1-\theta}$$

$$nr(1-\theta) = \theta \sum_{i=1}^n x_i$$

$$nr - nr\theta = \theta \sum_{i=1}^n x_i$$

$$nr = \theta\left(nr + \sum_{i=1}^n x_i\right)$$

**Estimador MLE:**
$$\hat{\theta}_{MLE} = \frac{nr}{nr + \sum_{i=1}^n x_i} = \frac{r}{r + \bar{x}}$$

### b) Aplicación con muestra específica

**Datos:** $n = 5$, $x = (3,0,2,17,9)$, $\sum x_i = 31$, $\bar{x} = 6.2$

$$\hat{\theta}_{MLE} = \frac{r}{r + 6.2}$$

**Para aplicar Jackknife, calculamos $\hat{\theta}_{(-i)}$ para cada observación:**

- $\hat{\theta}_{(-1)} = \frac{r}{r + \frac{31-3}{4}} = \frac{r}{r + 7}$
- $\hat{\theta}_{(-2)} = \frac{r}{r + \frac{31-0}{4}} = \frac{r}{r + 7.75}$
- $\hat{\theta}_{(-3)} = \frac{r}{r + \frac{31-2}{4}} = \frac{r}{r + 7.25}$
- $\hat{\theta}_{(-4)} = \frac{r}{r + \frac{31-17}{4}} = \frac{r}{r + 3.5}$
- $\hat{\theta}_{(-5)} = \frac{r}{r + \frac{31-9}{4}} = \frac{r}{r + 5.5}$

**Estimador Jackknife:**
$$\hat{\theta}_{Jack} = 5\hat{\theta}_{MLE} - 4 \cdot \frac{1}{5}\sum_{i=1}^5 \hat{\theta}_{(-i)}$$

---

## Ejercicio 3: Distribución Bernoulli

**Parámetro:** $\psi(\theta) = \theta(1-\theta)$ (varianza de Bernoulli)

### a) Estimador MLE de $\psi(\theta)$

Para Bernoulli: $\hat{\theta}_{MLE} = \bar{X}$

Por el método delta:
$$\hat{\psi}(\theta)_{MLE} = \psi(\hat{\theta}_{MLE}) = \bar{X}(1-\bar{X})$$

### b) Sesgo del estimador MLE

$$E[\hat{\psi}(\theta)_{MLE}] = E[\bar{X}(1-\bar{X})] = E[\bar{X}] - E[\bar{X}^2]$$

$$= \theta - (Var(\bar{X}) + (E[\bar{X}])^2)$$

$$= \theta - \left(\frac{\theta(1-\theta)}{n} + \theta^2\right)$$

$$= \theta - \theta^2 - \frac{\theta(1-\theta)}{n}$$

$$= \theta(1-\theta) - \frac{\theta(1-\theta)}{n} = \psi(\theta)\left(1 - \frac{1}{n}\right)$$

**Sesgo:** $Sesgo = -\frac{\psi(\theta)}{n}$

### c) Estimador Jackknife

$$\hat{\psi}(\theta)_{(-i)} = \bar{X}_{(-i)}(1-\bar{X}_{(-i)})$$

donde $\bar{X}_{(-i)} = \frac{n\bar{X} - X_i}{n-1}$

$$\hat{\psi}(\theta)_{Jack} = n\hat{\psi}(\theta)_{MLE} - (n-1)\frac{1}{n}\sum_{i=1}^n \hat{\psi}(\theta)_{(-i)}$$

### d) Sesgo del estimador Jackknife

El método Jackknife está diseñado para eliminar el sesgo de orden $O(1/n)$. 

Dado que el sesgo del estimador MLE es $-\frac{\psi(\theta)}{n}$, el estimador Jackknife tendrá sesgo de orden $O(1/n^2)$, que es asintóticamente despreciable.

**Resultado:** $Sesgo(\hat{\psi}(\theta)_{Jack}) = O(1/n^2) \approx 0$ para $n$ grande.
