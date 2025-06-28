# Estimador de Mínimos Cuadrados Generalizados para μ₁ - μ₂

## Planteamiento del Problema

Sean:
- X₁, ..., Xₙ una muestra aleatoria con distribución N(μ₁, σ²)
- Y₁, ..., Yₘ una muestra aleatoria con distribución N(μ₂, θσ²)
- Las muestras son independientes, μ₁, μ₂, σ² desconocidos, y θ conocido

**Objetivo:** Encontrar el estimador de mínimos cuadrados generalizados de μ₁ - μ₂ y su correspondiente intervalo de confianza.

## Desarrollo de la Solución

### Paso 1: Formulación del Sistema

Del problema planteado tenemos:
- Z₁ = X₁, i = 1, ..., n
- Zₙ₊ⱼ = Yⱼ, j = 1, ..., m

Por tanto, el vector de observaciones es:
$$\mathbf{Z} = \begin{pmatrix} Z_1 \\ \vdots \\ Z_n \\ Z_{n+1} \\ \vdots \\ Z_{n+m} \end{pmatrix} = \begin{pmatrix} X_1 \\ \vdots \\ X_n \\ Y_1 \\ \vdots \\ Y_m \end{pmatrix}$$

### Paso 2: Modelo Lineal

El modelo lineal es:
$$\mathbf{Z} = \mathbf{A}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

donde:
- $\boldsymbol{\beta} = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}$

- $\mathbf{A} = \begin{pmatrix} \mathbf{1}_n & \mathbf{0}_n \\ \mathbf{0}_m & \mathbf{1}_m \end{pmatrix}$ (matriz de diseño)

- $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{\Sigma})$

### Paso 3: Matriz de Covarianzas

La matriz de covarianzas Σ es:
$$\mathbf{\Sigma} = \begin{pmatrix} \mathbf{I}_n & \mathbf{0} \\ \mathbf{0} & \theta\mathbf{I}_m \end{pmatrix}$$

donde $\mathbf{I}_n$ e $\mathbf{I}_m$ son las matrices identidad de dimensiones n y m respectivamente.

### Paso 4: Estimador de Mínimos Cuadrados Generalizados

El estimador MCG está dado por:
$$\hat{\boldsymbol{\beta}} = (\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{Z}$$

**Cálculo de Σ⁻¹:**
$$\mathbf{\Sigma}^{-1} = \begin{pmatrix} \mathbf{I}_n & \mathbf{0} \\ \mathbf{0} & \frac{1}{\theta}\mathbf{I}_m \end{pmatrix}$$

**Cálculo de A^T Σ⁻¹:**
$$\mathbf{A}^T\mathbf{\Sigma}^{-1} = \begin{pmatrix} \mathbf{1}_n^T & \mathbf{0} \\ \mathbf{0} & \frac{1}{\theta}\mathbf{1}_m^T \end{pmatrix}$$

**Cálculo de A^T Σ⁻¹ A:**
$$\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{A} = \begin{pmatrix} n & 0 \\ 0 & \frac{m}{\theta} \end{pmatrix}$$

**Inversa:**
$$(\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{A})^{-1} = \begin{pmatrix} \frac{1}{n} & 0 \\ 0 & \frac{\theta}{m} \end{pmatrix}$$

### Paso 5: Estimadores de μ₁ y μ₂

$$\mathbf{A}^T\mathbf{\Sigma}^{-1}\mathbf{Z} = \begin{pmatrix} \sum_{i=1}^n X_i \\ \frac{1}{\theta}\sum_{j=1}^m Y_j \end{pmatrix}$$

Por tanto:
$$\hat{\boldsymbol{\beta}} = \begin{pmatrix} \hat{\mu}_1 \\ \hat{\mu}_2 \end{pmatrix} = \begin{pmatrix} \frac{1}{n}\sum_{i=1}^n X_i \\ \frac{1}{m}\sum_{j=1}^m Y_j \end{pmatrix} = \begin{pmatrix} \bar{X} \\ \bar{Y} \end{pmatrix}$$

### Paso 6: Estimador de μ₁ - μ₂

El estimador de μ₁ - μ₂ es:
$$\widehat{\mu_1 - \mu_2} = \hat{\mu}_1 - \hat{\mu}_2 = \bar{X} - \bar{Y}$$

### Paso 7: Varianza del Estimador

$$\text{Var}(\bar{X} - \bar{Y}) = \text{Var}(\bar{X}) + \text{Var}(\bar{Y}) = \frac{\sigma^2}{n} + \frac{\theta\sigma^2}{m} = \sigma^2\left(\frac{1}{n} + \frac{\theta}{m}\right)$$

### Paso 8: Estimación de σ²

Para construir el intervalo de confianza, necesitamos estimar σ². Usamos:
$$\hat{\sigma}^2 = \frac{\sum_{i=1}^n (X_i - \bar{X})^2 + \frac{1}{\theta}\sum_{j=1}^m (Y_j - \bar{Y})^2}{n + m - 2}$$

### Paso 9: Distribución del Estimador y Normalización

**Distribución de $\bar{X} - \bar{Y}$:**

Sabemos que:
- $\bar{X} \sim N\left(\mu_1, \frac{\sigma^2}{n}\right)$
- $\bar{Y} \sim N\left(\mu_2, \frac{\theta\sigma^2}{m}\right)$

Por independencia:
$\bar{X} - \bar{Y} \sim N\left(\mu_1 - \mu_2, \frac{\sigma^2}{n} + \frac{\theta\sigma^2}{m}\right)$

Es decir:
$\bar{X} - \bar{Y} \sim N\left(\mu_1 - \mu_2, \sigma^2\left(\frac{1}{n} + \frac{\theta}{m}\right)\right)$

**Paso 1: Centrar la distribución**
$(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2) \sim N\left(0, \sigma^2\left(\frac{1}{n} + \frac{\theta}{m}\right)\right)$

**Paso 2: Estandarizar (normalizar para varianza 1)**

Para obtener varianza 1, dividimos por la desviación estándar:

$\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sigma\sqrt{\frac{1}{n} + \frac{\theta}{m}}} \sim N(0,1)$

**Verificación de la normalización:**
- **Media:** $E\left[\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sigma\sqrt{\frac{1}{n} + \frac{\theta}{m}}}\right] = \frac{0}{\sigma\sqrt{\frac{1}{n} + \frac{\theta}{m}}} = 0$ ✓

- **Varianza:** $\text{Var}\left[\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sigma\sqrt{\frac{1}{n} + \frac{\theta}{m}}}\right] = \frac{\sigma^2\left(\frac{1}{n} + \frac{\theta}{m}\right)}{\sigma^2\left(\frac{1}{n} + \frac{\theta}{m}\right)} = 1$ ✓

**Paso 3: Reemplazar σ por su estimador**

Como σ es desconocido, lo reemplazamos por $\hat{\sigma}$:

$\frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\hat{\sigma}\sqrt{\frac{1}{n} + \frac{\theta}{m}}} \sim t_{n+m-2}$

**¿Por qué distribución t?**
- Al reemplazar σ por $\hat{\sigma}$, introducimos variabilidad adicional
- La distribución t captura esta incertidumbre extra
- Los grados de libertad son n+m-2 porque estimamos 2 parámetros (μ₁ y μ₂)

### Paso 10: Intervalo de Confianza

El intervalo de confianza al (1-α)100% para μ₁ - μ₂ es:

$$(\bar{X} - \bar{Y}) \pm t_{\alpha/2, n+m-2} \cdot \hat{\sigma}\sqrt{\frac{1}{n} + \frac{\theta}{m}}$$

## Resumen Final

- **Estimador MCG:** $\widehat{\mu_1 - \mu_2} = \bar{X} - \bar{Y}$
- **Varianza estimada:** $\widehat{\text{Var}}(\bar{X} - \bar{Y}) = \hat{\sigma}^2\left(\frac{1}{n} + \frac{\theta}{m}\right)$
- **Intervalo de confianza:** $(\bar{X} - \bar{Y}) \pm t_{\alpha/2, n+m-2} \cdot \hat{\sigma}\sqrt{\frac{1}{n} + \frac{\theta}{m}}$

donde $\hat{\sigma}^2 = \frac{\sum_{i=1}^n (X_i - \bar{X})^2 + \frac{1}{\theta}\sum_{j=1}^m (Y_j - \bar{Y})^2}{n + m - 2}$
