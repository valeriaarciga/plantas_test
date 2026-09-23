# Formulación: optimización de la plantación en reforestación sobre retículos hexagonales

**Versión ejecutable — alcance de licenciatura avanzado**
Caso: Línea de Transmisión Dominica–Charcas (Altiplano Mexicano, zona semiárida)

---

## 0. Alcance: qué se modela y qué no

Declarar el alcance explícitamente es parte del rigor, y evita que el jurado pregunte por algo que se omitió sin justificación.

**Sí se modela:**
- Asignación espacial de 10 especies sobre el retículo tres bolillos, minimizando competencia y monocultivo.
- Vegetación preexistente incierta, mediante simulación de Montecarlo sobre escenarios independientes.
- Cantidad de planta a adquirir por especie, con garantía probabilística de supervivencia.
- Cotas exactas de optimalidad vía MILP y comparación justa contra el metaheurístico.

**No se modela (declarado como trabajo futuro):**
- *Recurso de dos etapas.* La compra se calcula de forma analítica y desacoplada, no como variable de primera etapa con recurso. Consecuencia honesta: al resolver cada escenario por separado se obtiene $\mathbb{E}_\omega[F^\star(\omega)]$, que es una **cota inferior** del óptimo de un modelo de dos etapas. Se reporta como tal.
- *Remoción de planta preexistente.* Los nodos ocupados quedan fijos.
- *Dinámica temporal.* Se optimiza la configuración de plantación, no una trayectoria a 5 años.

---

## 1. El grafo del terreno

El patrón **"tres bolillos"** es un empaquetamiento hexagonal; su **grafo de adyacencia es el retículo triangular**, 6-regular en el interior. Con filas alternadamente desplazadas, el vecindario de $v=(r,c)$ es:

$$
\mathcal{N}(r,c) = \{(r,c{\pm}1)\} \cup \{(r{\pm}1, c'),\; c' \in \begin{cases}\{c{-}1,c\} & r \text{ par}\\ \{c,c{+}1\} & r \text{ impar}\end{cases}\}
$$

**Escala real (1 ha):** $R=14$, $C=47$, $|V|=658$, $|A|=1853$, grado medio 5.63.

### 1.1 Dos propiedades que se usan como herramienta (verificadas)

**Propiedad A — 3-coloración.** Con coordenadas axiales $q = c - \lfloor(r-(r\bmod 2))/2\rfloor$, el color $\kappa(v)=(q-r)\bmod 3$ es una coloración propia: **0 aristas monocromáticas**, clases de tamaño $\{224,217,217\}$. Cada clase es un conjunto independiente. Se verificó además $\alpha(G)=224$ resolviendo el máximo conjunto independiente por programación entera.

**Propiedad B — El mínimo de monocultivo es cero, y se construye.** Si cada especie se aloja dentro de una sola clase de color, ningún par de vecinos comparte especie. Resolviendo el empaquetamiento $\sum_k n_{ik}=R_i$, $\sum_i n_{ik}=|K_k|$ se obtiene solución factible **incluso con tolerancia $t=0$**:

| Clase (cap.) | Composición |
|---|---|
| $K_0$ (224) | AL 12, ASc 42, ASt 42, OE 38, OS 64, YF 26 |
| $K_1$ (217) | AL 30, AS 187 |
| $K_2$ (217) | AS 9, OC 49, OR 73, PL 86 |

Esto da dos cosas gratis: **un extremo del frente de Pareto conocido analíticamente** ($f_2^\star=0$) y **una solución inicial de altísima calidad** para el algoritmo evolutivo, construible en $O(|V|)$.

---

## 2. Conjuntos

| Símbolo | Definición | Tamaño |
|---|---|---|
| $V$ | Nodos (sitios de plantación) | 658 |
| $A$ | Aristas: pares de vecinos en competencia | 1853 |
| $S$ | Especies nativas | 10 |
| $\Omega$ | Escenarios Montecarlo de vegetación preexistente | 30 |
| $P^\omega\subset V$ | Nodos ocupados por planta preexistente en $\omega$ | ≈129 |
| $K_0,K_1,K_2$ | Clases independientes de la 3-coloración | 224, 217, 217 |

---

## 3. Parámetros

### 3.1 Tabla maestra

| $i$ | Especie | $R_i$ | $S_i$ | $\pi_i$ | $h_i$ (m) | Gremio | $\rho_i$ |
|---|---|---|---|---|---|---|---|
| AL | *Agave lechuguilla* | 42 | 0.9265 | 0.0633 | 0.4 | CAM-roseta | 0 |
| AS | *Agave salmiana* | 196 | 0.9082 | 0.2951 | 2.0 | CAM-roseta | 0 |
| ASc | *Agave scabra* | 42 | 0.7453 | 0.0620 | 1.0 | CAM-roseta | 0 |
| ASt | *Agave striata* | 42 | 0.9138 | 0.0596 | 0.6 | CAM-roseta | 0 |
| OC | *Opuntia cantabrigiensis* | 49 | 0.8287 | 0.0777 | 1.5 | CAM-cladodio | 0 |
| OE | *Opuntia engelmannii* | 38 | 0.8194 | 0.0619 | 2.0 | CAM-cladodio | 0 |
| OR | *Opuntia robusta* | 73 | 0.9198 | 0.1165 | 2.0 | CAM-cladodio | 0 |
| OS | *Opuntia streptacantha* | 64 | 0.8525 | 0.0971 | 3.5 | CAM-cladodio | 0 |
| PL | *Prosopis laevigata* | 86 | 0.8179 | 0.1280 | 8.0 | C3-leñosa | 1 |
| YF | *Yucca filifera* | 26 | 0.7263 | 0.0387 | 8.0 | CAM-arborescente | 0.5 |

Otros: $N=658$, $p_{\text{ocu}}=0.19633$, $t\in[0,0.10]$.

$S_i$, $R_i$, $\pi_i$, $p_{\text{ocu}}$ provienen de datos de campo validados. **Las alturas $h_i$ son valores de literatura general y son el parámetro más débil**: el socio formador solo reportó tallas de vivero (15–50 cm), inútiles a horizonte de 5 años. Por eso $h_i$ entra al análisis de sensibilidad.

### 3.2 Cuotas reescaladas para instancias reducidas

**Detalle práctico crítico.** Las cuotas $R_i$ están definidas para $|V|=658$. Al probar el MILP en retículas pequeñas hay que reescalar, o el modelo es infactible:

$$
R_i^{(V)} = \text{redondeo por mayores residuos}\left(\frac{R_i}{658}\,|V|\right), \qquad \text{con } \sum_i R_i^{(V)} = |V|
$$

Con $|V|=16$ hay especies que reciben 1 planta y el problema se vuelve degenerado; **se recomienda empezar la escalera en $6\times6$ ($|V|=36$)**.

---

## 4. Matriz de competencia basada en rasgos

En vez de asignar 55 valores a mano, se derivan de tres rasgos con tres hiperparámetros auditables.

Con altura normalizada en log, $\hat h_i=\frac{\ln h_i-\min\ln h}{\max\ln h-\min\ln h}$:

$$
d_{ij}^2 = w_h(\hat h_i-\hat h_j)^2 + w_\rho(\rho_i-\rho_j)^2 + w_g\,\mathbb 1[g_i\ne g_j]
$$

$$
\theta_{ij}=\exp\!\Big(-\tfrac{d_{ij}^2}{2\sigma^2}\Big), \qquad
\boxed{C_{ij}=\theta_{ij}-\beta F_{ij}}, \qquad
\boxed{\tilde C_{ij}=S_iS_j\,C_{ij}}
$$

donde $F_{ij}=1$ si el par une a *Prosopis* (nodriza) con una suculenta.

**Justificación (3 referencias bastan):** Kunstler et al. (2016, *Nature*) muestran que los rasgos funcionales predicen competencia de forma globalmente consistente; Chesson (2000) fundamenta que la competencia crece con el solapamiento de nicho; Padilla & Pugnaire (2006) documentan a *Prosopis* como planta nodriza en ambientes áridos, lo que justifica $C_{ij}<0$.

**Propiedades:** $C_{ii}=1$ sale automáticamente (solapamiento total), no por decreto. La ponderación $\tilde C_{ij}=S_iS_j C_{ij}$ es la **competencia esperada realizada** (ambos individuos deben sobrevivir para competir); sustituye al factor $\frac{1}{1-S_iS_j}$, que crecía con la supervivencia —penalizando plantar especies robustas juntas, lo opuesto a lo deseable— y divergía cuando $S_iS_j\to1$.

Con $\sigma{=}0.5,\ \beta{=}0.20,\ \mathbf w{=}(0.4,0.4,0.2)$, valores representativos: $C_{\text{OE,OR}}=1.000$, $C_{\text{AL,ASt}}=0.985$, $C_{\text{AS,ASc}}=0.958$, $C_{\text{PL,AL}}=-0.065$, $C_{\text{PL,OC}}=0.035$. La estructura emergente es cualitativamente distinta de la matriz por morfología: la columna de *Prosopis* es casi nula o negativa (separación de nicho vertical + facilitación).

**Sensibilidad reducida a 3 corridas:** $\beta\in\{0,\,0.20,\,0.40\}$ con $\sigma,\mathbf w$ fijos. Suficiente para mostrar si el ranking de soluciones es estable.

---

## 5. Variables de decisión

$$
x_{v,i}=\begin{cases}1&\text{si el nodo }v\text{ recibe la especie }i\\0&\text{si no}\end{cases}
\qquad
y_{a,i,j}\ge 0 \;\;\text{(linealiza } x_{u,i}x_{v,j}\text{, } a=\{u,v\})
$$

---

## 6. Funciones objetivo

Ambas **lineales en $y$** y por tanto **idénticas en el MILP y en el metaheurístico**. Este es el requisito que hace válida toda comparación posterior, y es exactamente lo que faltaba en la versión previa del trabajo.

$$
f_1=\sum_{a=\{u,v\}\in A}\sum_{i\in S}\sum_{j\in S}\tilde C_{ij}\,y_{a,i,j}
\qquad\text{(competencia esperada)}
$$

$$
f_2=\sum_{a\in A}\sum_{i\in S} y_{a,i,i}
\qquad\text{(aristas monoespecíficas: monocultivo)}
$$

**Por qué $f_2$ así y no como equidad composicional.** Medir diversidad como $\sum_i|n_i/N-1/|S||$ tiene un piso teórico inalcanzable de 0.386–0.431, porque la cuota obliga a *Agave salmiana* a ~30 % y el objetivo empuja hacia 10 %. Al medir **mezcla espacial** en vez de equidad, el óptimo es 0 y es alcanzable respetando la cuota exacta (Propiedad B). Además responde literalmente al requisito "evitar monocultivos" del reto, y no cuesta variables extra: usa las mismas $y$ que $f_1$.

Los objetivos son genuinamente conflictivos: $f_1$ quiere separar especies funcionalmente parecidas (dos *Agave*, $\tilde C\approx0.83$), mientras $f_2$ solo penaliza pares idénticos; bajar uno puede forzar el otro.

**Métricas reportadas (no optimizadas):** índice de mezcla de Gadow $M=\frac{1}{|V|}\sum_v\frac{1}{|\mathcal N(v)|}\sum_{u\in\mathcal N(v)}\mathbb 1[s_u\ne s_v]$ y divergencia de cuota $D_{KL}(p\Vert r)=\sum_i p_i\ln(p_i/r_i)$.

---

## 7. Modelo MILP (por escenario)

$$
\min\;\; \lambda_1 f_1+\lambda_2 f_2
$$

sujeto a:

**(C1) Asignación única.** $\displaystyle\sum_{i\in S}x_{v,i}=1\quad \forall v\in V$

**(C2) Linealización de Frieze–Yadegar.** Reemplaza el producto $x_{u,i}x_{v,j}$ **sin exigir que $y$ sea binaria**, lo que reduce fuertemente el árbol de ramificación:
$$\sum_{j\in S}y_{a,i,j}=x_{u,i},\qquad \sum_{i\in S}y_{a,i,j}=x_{v,j},\qquad y_{a,i,j}\ge0 \qquad \forall a=\{u,v\}\in A$$

**(C3) Cuotas con tolerancia.** $\displaystyle\big\lceil(1-t)R^{(V)}_i\big\rceil\le\sum_{v\in V}x_{v,i}\le\big\lfloor(1+t)R^{(V)}_i\big\rfloor \quad\forall i$

**(C4) Vegetación preexistente fija.** $x_{v,s^\omega_v}=1\quad\forall v\in P^\omega$

**(C5) Dominios.** $x_{v,i}\in\{0,1\}$, $y_{a,i,j}\ge0$

Cinco familias de restricciones. Es un modelo que se escribe en Gurobi en ~40 líneas.

### 7.1 Frente de Pareto exacto por $\varepsilon$-restricción

$$
\min f_1 \quad\text{s.a.}\quad f_2\le\varepsilon,\quad \text{(C1)–(C5)}, \qquad \varepsilon=0,1,2,\dots,\varepsilon_{\max}
$$

Como $f_2$ es **entera y de rango pequeño** (número de aristas monoespecíficas), el barrido es natural y no requiere calibrar una malla continua. Con 8–10 valores de $\varepsilon$ se obtiene el frente exacto.

### 7.2 Tamaño y argumento de intratabilidad

| Instancia | $\|V\|$ | $\|A\|$ | Binarias $x$ | Continuas $y$ | Restricciones (C2) |
|---|---|---|---|---|---|
| 4×4 | 16 | 33 | 160 | 3 300 | 660 |
| 6×6 | 36 | 85 | 360 | 8 500 | 1 700 |
| 8×8 | 64 | 161 | 640 | 16 100 | 3 220 |
| 10×10 | 100 | 261 | 1 000 | 26 100 | 5 220 |
| 12×12 | 144 | 385 | 1 440 | 38 500 | 7 700 |
| **14×47 (real)** | **658** | **1 853** | **6 580** | **185 300** | **37 060** |

El argumento es **estructural y citable**: el problema es un *Quadratic Assignment Problem* (Koopmans & Beckmann, 1957) con 658 elementos, mientras que las mejores resoluciones exactas publicadas de QAP alcanzan $n\approx30$. La curva empírica de tiempos documenta el crecimiento.

---

## 8. Simulación de Montecarlo

Para cada $\omega\in\Omega$:
1. **Ocupación:** $\mathbb 1[v\in P^\omega]\sim\text{Bernoulli}(0.19633)$ para cada $v\in V$.
2. **Especie:** $s^\omega_v\sim\text{Categórica}(\pi_1,\dots,\pi_{10})$ para cada $v\in P^\omega$.

Cada escenario se optimiza **de forma independiente**; se reporta la distribución de $(f_1^\star,f_2^\star)$ (media, desviación, percentiles 5 y 95), no un valor puntual. Los escenarios se generan una sola vez, se guardan en disco y se reutilizan en MILP y metaheurístico con las mismas semillas.

**Resultado analítico de validación.** Cuando dos nodos vecinos están *ambos* ocupados por preexistentes de la misma especie, esa arista monoespecífica es inevitable. Su valor esperado es

$$
\mathbb E[f_2^{\min}]=|A|\,p_{\text{ocu}}^2\sum_i\pi_i^2=1853\times0.19633^2\times0.14928=10.66
$$

Simulación con 2 000 réplicas: media 10.62, desviación 3.70, $[P_5,P_{95}]=[5,17]$. **Este número sirve como prueba de correctitud del código**: si el optimizador reporta $f_2$ por debajo de esta cota, hay un error de implementación. Y es el argumento sustantivo de por qué la incertidumbre importa: el óptimo de $f_2$ pasa de 0 (terreno limpio) a una variable aleatoria de media ≈10.6.

**Tamaño muestral.** Con $\sigma_{f_2}\approx3.7$, el error estándar con $|\Omega|=30$ es $0.68$ aristas — suficiente para conclusiones preliminares. Se usa $|\Omega|=5$ en el MILP (por tratabilidad) y $|\Omega|=30$ en el metaheurístico.

---

## 9. Módulo de compra con garantía probabilística

Desacoplado del modelo espacial, resuelto en forma cerrada. Sea $n_i^{\min}(\alpha)$ el mínimo de individuos a plantar tal que la probabilidad de tener al menos $R_i$ sobrevivientes sea $\ge 1-\alpha$:

$$
n_i^{\min}(\alpha)=\min\Big\{n\in\mathbb Z_+ : \Pr[\mathrm{Bin}(n,S_i)\ge R_i]\ge1-\alpha\Big\}
$$

Como la condición es monótona en $n$, se calcula **exactamente** con la CDF binomial (`scipy.stats.binom.sf`), en un bucle de tres líneas. No requiere aproximación normal ni optimización.

| Especie | $R_i$ | $S_i$ | $\alpha{=}0.30$ | $\alpha{=}0.20$ | $\alpha{=}0.10$ | $\alpha{=}0.05$ |
|---|---|---|---|---|---|---|
| AL | 42 | 0.9265 | 46 | 47 | 48 | 49 |
| AS | 196 | 0.9082 | 218 | 220 | 222 | 224 |
| ASc | 42 | 0.7453 | 58 | 60 | 62 | 64 |
| ASt | 42 | 0.9138 | 47 | 48 | 49 | 50 |
| OC | 49 | 0.8287 | 61 | 62 | 64 | 65 |
| OE | 38 | 0.8194 | 48 | 49 | 51 | 52 |
| OR | 73 | 0.9198 | 81 | 82 | 83 | 84 |
| OS | 64 | 0.8525 | 77 | 78 | 80 | 81 |
| PL | 86 | 0.8179 | 108 | 109 | 111 | 113 |
| YF | 26 | 0.7263 | 37 | 39 | 41 | 42 |
| **Total** | **658** | | **781** | **794** | **811** | **824** |
| Sobrecompra | | | +18.7 % | +20.7 % | +23.3 % | +25.2 % |

**Resultado de valor práctico:** la sobrecompra no es uniforme. *Yucca filifera* requiere +50 % y *Agave scabra* +43 %, mientras *Opuntia robusta* solo +12 %. Un margen plano del 10 % —práctica habitual— dejaría a esas dos especies sistemáticamente por debajo de la meta contractual de supervivencia (70–80 % exigido por el socio formador).

Este módulo por sí solo es un entregable completo, y se produce en media hora.

---

## 10. Algoritmo evolutivo multiobjetivo adaptado al retículo

### 10.1 Representación
$\mathbf s\in\{1,\dots,10\}^{|V|}$, un entero por nodo (658 enteros, no 6 580 binarias). Nodos de $P^\omega$ marcados como congelados.

### 10.2 Operadores adaptados a la geometría (la contribución algorítmica)

**(a) Inicialización por 3-coloración.** La población arranca desde la partición de la Propiedad B: cada especie alojada en una clase independiente. Toda la población nace con $f_2$ en su mínimo alcanzable, es decir, **desde un extremo conocido del frente de Pareto**. Se perturba con $\mu$ intercambios aleatorios para dar diversidad.

**(b) Mutación por intercambio (swap).** Se eligen dos nodos libres con especies distintas y se permutan. **Propiedad clave: el intercambio preserva exactamente los conteos por especie**, luego la restricción de cuota (C3) es invariante y *no se necesita operador de reparación*. Esta es la principal fuente de velocidad frente al enfoque previo.

**(c) Cruzamiento por bola hexagonal.** Se elige un centro $c$ y un radio $\varrho$, y se hereda del segundo padre $B(c,\varrho)=\{v: d_G(v,c)\le\varrho\}$. Preserva patrones espaciales compactos, a diferencia de partir por filas, que rompe la vecindad del retículo triangular. Requiere reparación por conteo tras el cruce.

**(d) Evaluación incremental.** Al intercambiar $u\leftrightarrow v$ solo cambian las aristas incidentes ($\le12$ de 1 853):
$$\Delta f_1=\!\!\sum_{w\in\mathcal N(u)\setminus\{v\}}\!\!\big(\tilde C_{s_v,s_w}-\tilde C_{s_u,s_w}\big)+\!\!\sum_{w\in\mathcal N(v)\setminus\{u\}}\!\!\big(\tilde C_{s_u,s_w}-\tilde C_{s_v,s_w}\big)$$
Costo $O(12)$ en lugar de $O(1853)$: aceleración ≈150×. *(Opcional, solo si sobra tiempo: usarla en una búsqueda local sobre los nodos más conflictivos, convirtiendo el GA en memético.)*

**(e) Selección.** NSGA-II estándar (dominancia + *crowding distance*). Con dos objetivos NSGA-II es apropiado; la migración a NSGA-III solo se justificaría con ≥4 objetivos.

### 10.3 Presupuesto computacional (medido en la implementación de referencia)

| Operación | Tiempo medido |
|---|---|
| Evaluación completa $(f_1,f_2)$ en 14×47 | **24.5 µs** |
| Pob. 200 × 200 gen., 1 escenario | ~1 s |
| Pob. 200 × 200 gen., 30 escenarios | ~30 s |
| Búsqueda local, 40 000 intercambios | ~20 s |

La escala real es holgadamente tratable para el metaheurístico, en contraste directo con el MILP.

### 10.4 Hallazgo: la siembra por coloración es necesaria pero no suficiente

En **terreno limpio**, la siembra por 3-coloración seguida de búsqueda local alcanza $f_2 = 0$ exactamente, confirmando la Propiedad B de forma constructiva, y reduce $f_1$ de 765.8 a 673.6.

Con **vegetación preexistente**, la siembra por sí sola es insuficiente: los nodos fijos caen dispersos en las tres clases de color y rompen la estructura. En un escenario representativo (piso teórico $f_2^{\min}=4$), la siembra da $f_2=103$ y solo tras la búsqueda local baja a 29. **Conclusión de diseño: la búsqueda local no es un refinamiento opcional sino un componente necesario del algoritmo**, y el hueco residual entre 29 y el piso de 4 es precisamente el margen que el metaheurístico debe cerrar y que el MILP permitirá acotar en instancias pequeñas.

---

## 11. Protocolo de comparación justa

1. **Mismo vector objetivo** $(f_1,f_2)$, misma $\tilde C$, misma $t$, mismos escenarios con las mismas semillas.
2. **Frente contra frente:** MILP produce el frente exacto por $\varepsilon$-restricción; NSGA-II el aproximado.
3. **Métricas:** **hipervolumen** (único indicador unario estrictamente Pareto-compatible) con punto de referencia común, más **gap por objetivo** en los extremos lexicográficos. `pymoo.indicators.hv` lo da en una línea.
4. **Robustez:** ≥10 semillas por configuración; medianas e intervalos intercuartílicos, nunca una sola corrida.
5. **Escalabilidad:** tiempo a optimalidad (MILP) y tiempo a estancamiento de HV (NSGA-II) por tamaño.

---

## 12. Plan de experimentos (sábado tarde → lunes)

Cada bloque entrega algo publicable aunque el siguiente no salga. Los bloques ★ son el mínimo para tener resultados el lunes.

| # | Bloque | Horas | Entregable |
|---|---|---|---|
| ★1 | **Núcleo** | 2 | Grafo, 3-coloración, matriz $C$, escenarios, objetivos vectorizados. *Test: reproducir $\|A\|{=}1853$, $\alpha{=}224$, $f_2{=}0$ en la solución sembrada, $\mathbb E[f_2^{\min}]{=}10.6$.* |
| ★2 | **Compra** | 0.5 | Tabla del §9 completa. Entregable independiente y cerrado. |
| ★3 | **MILP pequeño** | 3 | Gurobi, 6×6 y 8×8, 5 escenarios, barrido $\varepsilon$ de 8 puntos → frente exacto + tiempos. |
| ★4 | **NSGA-II** | 3 | pymoo con los operadores de §10.2. Validación contra el frente exacto en 6×6 y 8×8 (HV, gap). |
| ★5 | **Escalabilidad** | 2 | MILP en 10×10 y 12×12 con límite de 3 600 s registrando el **MIP gap aunque no cierre** — el gap abierto *es* el resultado. NSGA-II en 14×47. Curva log-lineal. |
| 6 | **Montecarlo completo** | 1.5 | NSGA-II con $\|\Omega\|{=}30$ en escala real; distribución de $(f_1^\star,f_2^\star)$; validación contra la cota 10.6. |
| 7 | **Sensibilidad** | 1 | $\beta\in\{0,0.2,0.4\}$; estabilidad del ranking. |
| 8 | **Redacción y figuras** | resto | Frente exacto vs aproximado; mapa de la solución compromiso; curva de escalabilidad; histograma de $f_2^{\min}$. |

**Trucos de implementación que ahorran horas:**
- Aristas como dos arreglos NumPy `U`, `V` precalculados. Nunca recorrer el grafo con bucles de Python dentro de la función objetivo.
- Escenarios congelados en un `.npz`, compartidos por MILP y GA.
- En Gurobi declarar $y$ como `CONTINUOUS`: Frieze–Yadegar no requiere integralidad y el modelo se resuelve mucho más rápido.
- `MIPFocus=1`, `TimeLimit=3600`, y **guardar siempre `model.MIPGap`**.
- Fijar $x_{v,i}$ de los nodos preexistentes con `.lb = .ub = 1`: el presolve reduce el modelo notablemente.

---

## 13. Resumen

Se formula la planificación de la reforestación como un **problema de asignación cuadrática biobjetivo sobre el retículo triangular inducido por el patrón tres bolillos**, con cuotas por especie como restricción dura e incertidumbre de la vegetación preexistente integrada por simulación de Montecarlo.

Tres elementos distinguen la propuesta: (i) la **matriz de competencia se deriva de rasgos funcionales** —altura, profundidad radicular y gremio fotosintético— con un término de facilitación para *Prosopis*, en lugar de asignarse a mano, lo que la hace auditable y sujeta a análisis de sensibilidad; (ii) la **diversidad se redefine como propiedad espacial** (aristas monoespecíficas), eliminando el piso teórico inalcanzable de 0.386 que padecía la medida por equidad composicional; y (iii) se aprovecha la **3-coloración del retículo triangular** para demostrar constructivamente que $f_2^\star=0$ es alcanzable con cuota exacta y para sembrar la población del algoritmo evolutivo en ese extremo del frente.

El MILP, linealizado con el esquema de Frieze–Yadegar, produce frentes de Pareto exactos por $\varepsilon$-restricción en instancias reducidas; su crecimiento de 3 300 a 185 300 variables entre 4×4 y la escala real documenta cuantitativamente la intratabilidad, coherente con que el problema es un QAP de 658 elementos frente a un límite práctico de resolución exacta de $n\approx30$. El algoritmo evolutivo, con mutación por intercambio que preserva cuotas por construcción y cruzamiento por bola hexagonal, resuelve la escala real en segundos y se valida contra el frente exacto mediante hipervolumen, ahora sí **sobre la misma función objetivo en ambos métodos** — corrigiendo el defecto metodológico central de la formulación previa.

---

## Referencias

- Chesson, P. (2000). Mechanisms of maintenance of species diversity. *Annual Review of Ecology and Systematics*, 31, 343–366.
- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evolutionary Computation*, 6(2), 182–197.
- Frieze, A. M. & Yadegar, J. (1983). On the quadratic assignment problem. *Discrete Applied Mathematics*, 5(1), 89–98.
- Haimes, Y., Lasdon, L. & Wismer, D. (1971). On a bicriterion formulation... (método de $\varepsilon$-restricción). *IEEE Trans. Systems, Man and Cybernetics*, 1(3), 296–297.
- Koopmans, T. C. & Beckmann, M. (1957). Assignment problems and the location of economic activities. *Econometrica*, 25(1), 53–76.
- Kunstler, G. et al. (2016). Plant functional traits have globally consistent effects on competition. *Nature*, 529, 204–207.
- Padilla, F. M. & Pugnaire, F. I. (2006). The role of nurse plants in the restoration of degraded environments. *Frontiers in Ecology and the Environment*, 4(4), 196–202.
- Zitzler, E. & Thiele, L. (1999). Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach. *IEEE Trans. Evolutionary Computation*, 3(4), 257–271.
