# 🛒 E-Commerce Customer Analytics

Pipeline de análisis end-to-end sobre datos de clientes de e-commerce: desde la normalización de CSVs crudos hasta un modelo predictivo de churn con segmentación de compradores.

---

## 📋 Descripción del proyecto

Este proyecto construye un pipeline de datos completo que cubre las etapas fundamentales de un proyecto de analytics real:

1. **Modelado de base de datos** en esquema copo de nieve (Snowflake Schema)
2. **ETL** desde CSVs crudos de Kaggle hacia SQLite
3. **Análisis exploratorio** (EDA) con estadísticas descriptivas y correlaciones
4. **Segmentación de clientes** con K-Means (Buyer Personas)
5. **Modelo predictivo de churn** con Logistic Regression y XGBoost
6. **Análisis de cierre** cruzando segmentos con scores de riesgo

> **Nota sobre los datos:** Los datasets son sintéticos obtenidos de Kaggle. Los scores del modelo son excepcionalmente altos (AUC ~0.99) precisamente porque las variables fueron construidas para predecir el churn — algo esperado y documentado en el análisis.

---

## 🗂️ Estructura del repositorio

```
ecommerce_db/
│
├── config.py                  # Rutas y constantes centralizadas
│
├── 01_create_schema.py        # DDL: crea tablas, índices y vistas en SQLite
├── 02_load_customers.py       # ETL: carga clientes (features + targets)
├── 03_load_orders.py          # ETL: carga órdenes y dimensiones de fecha
├── 04_validate.py             # Validación de integridad referencial
│
├── 05_eda.py                  # Análisis exploratorio (4 secciones)
├── 06_segmentation.py         # K-Means: segmentación de clientes
├── 07_churn_model.py          # Modelos predictivos de churn
├── 08_risk_analysis.py        # Análisis de cierre: segmentos × riesgo
│
├── run_all.py                 # Orquestador: corre el pipeline completo
│
└── snowflake_schema_ddl.sql   # DDL standalone para referencia
```

---

## 🗄️ Esquema de base de datos

El proyecto implementa un **Snowflake Schema** con dos subject areas independientes:

```
dim_loyalty_status ──► fact_customers
                           (6,000 clientes · churn label · métricas de comportamiento)

dim_date ────────────┐
dim_product_category─┤
dim_payment_method ──┼──► fact_orders
dim_order_status ────┘        (2,600 órdenes · valor · rating · estado)

── Tablas generadas por el pipeline ──
customer_segments   (K-Means: segmento asignado por cliente)
churn_scores        (score 0–1 · tier Bajo/Medio/Alto/Crítico)
```

Los dos bloques no comparten llaves — limitación conocida del dataset fuente, documentada en el análisis.

---

## 📊 Resultados principales

### Segmentación de clientes (K-Means, k=4)

| Segmento | Clientes | Churn rate | Perfil |
|---|---|---|---|
| 💎 Cliente Premium | 1,845 (30.8%) | 0.2% | Alto valor, alta frecuencia, muy comprometido |
| 🏷️ Cazador de Ofertas | 1,928 (32.1%) | 1.1% | Compra por descuento, baja lealtad intrínseca |
| 🌟 Cliente Leal | 1,362 (22.7%) | 14.4% | Buen engagement, en zona intermedia de riesgo |
| ⚠️ Cliente en Riesgo | 865 (14.4%) | 82.0% | Alta inactividad, prácticamente perdidos |

### Modelo predictivo de churn

| Modelo | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.9608 | **0.9247** | 0.8798 | **0.9931** |
| XGBoost | 0.9592 | 0.9032 | 0.8727 | 0.9925 |

**Ganador: Logistic Regression** — mejor recall y AUC. En churn prediction el recall es la métrica crítica: minimizar los falsos negativos (clientes que se van sin ser detectados).

**Variables más importantes:** `days_since_last_purchase` (+305% diferencia entre grupos) y `engagement_score` (5.35 activos vs 2.36 churned).

### Análisis de cierre: Críticos por segmento

| Segmento | Críticos | % del segmento | Acción recomendada |
|---|---|---|---|
| ⚠️ Cliente en Riesgo | 703 | 81.3% | Ya perdidos — bajo ROI de intervención |
| 🌟 Cliente Leal | 179 | 13.1% | **Prioridad de retención** |
| 🏷️ Cazador de Ofertas | 11 | 0.6% | Reactivar con oferta puntual |
| 💎 Cliente Premium | 1 | 0.1% | Sin acción necesaria |

> Los **179 Clientes Leales en riesgo crítico** son el target real de retención — tienen historial de valor y todavía son recuperables.

---

## ⚙️ Instalación y uso

### Requisitos

- Python 3.10+
- Las librerías del proyecto (ver abajo)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ecommerce-customer-analytics.git
cd ecommerce-customer-analytics
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar rutas de los CSVs

Edita `config.py` y ajusta las rutas a los archivos fuente:

```python
CSV_FEATURES = Path("ruta/a/ecommerce_customer_features.csv")
CSV_TARGETS  = Path("ruta/a/ecommerce_customer_targets.csv")
CSV_ORDERS   = Path("ruta/a/daily_ecommerce_orders.csv")
```

> ⚠️ En Windows usa barras hacia adelante (`/`) o raw strings (`r"C:\ruta\..."`) para evitar errores de encoding.

### 4. Correr el pipeline

**Opción A — Script por script (recomendado la primera vez):**

```bash
python 01_create_schema.py   # Crea el esquema en SQLite
python 02_load_customers.py  # Carga clientes
python 03_load_orders.py     # Carga órdenes
python 04_validate.py        # Valida integridad

python 05_eda.py             # Análisis exploratorio
python 06_segmentation.py    # Segmentación K-Means
python 07_churn_model.py     # Modelo de churn
python 08_risk_analysis.py   # Análisis de cierre
```

**Opción B — Pipeline completo de una vez:**

```bash
python run_all.py
```

---

## 📦 Dependencias

```
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```

Instala todo con:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

## 🔍 Decisiones técnicas destacadas

**¿Por qué Snowflake Schema y no Star Schema?**
Las dimensiones categóricas de órdenes (categoría, método de pago, estado) tienen pocos valores únicos pero se normalizaron igual para demostrar el patrón completo. En producción con dimensiones más grandes la diferencia de performance sería significativa.

**¿Por qué SMOTE?**
El dataset tiene desbalance de clases (84.5% activos vs 15.5% churned). Sin balanceo, un modelo que prediga siempre "activo" alcanza 84.5% de accuracy trivialmente. SMOTE genera ejemplos sintéticos de la clase minoritaria solo en el set de entrenamiento.

**¿Por qué k=4 si el silhouette score óptimo es k=2?**
Los datos sintéticos no tienen clusters naturalmente separados (silhouette máximo: 0.1759). Se forzó k=4 por valor narrativo del portafolio, documentando explícitamente la decisión — exactamente lo que se haría en un contexto profesional cuando el negocio requiere segmentos accionables.

**¿Por qué ganó Logistic Regression sobre XGBoost?**
En datasets con separación lineal clara entre clases, LR es difícil de superar. XGBoost añade complejidad que no siempre se traduce en mejora cuando el patrón subyacente es relativamente simple.

---

## 📁 Datos fuente

Los CSVs utilizados provienen de Kaggle y no se incluyen en este repositorio por tamaño. Puedes encontrarlos buscando:

- `ecommerce customer churn dataset` — para los archivos de clientes
- `daily ecommerce orders dataset` — para el archivo de órdenes

---
