"""
07_churn_model.py — Modelo Predictivo de Churn
===============================================
Pasos:
  1. Cargar datos + feature engineering
  2. Split train/test + balanceo de clases (SMOTE)
  3. Entrenar Logistic Regression (baseline)
  4. Entrenar XGBoost (modelo final)
  5. Comparar métricas: accuracy, precision, recall, F1, ROC-AUC
  6. Visualizaciones: ROC, Confusion Matrix, Feature Importance, Score por segmento
  7. Persistir churn_score por cliente en la DB

Prerequisito : haber corrido 01–06 (DB con customer_segments lista)
Dependencias : pip install xgboost scikit-learn imbalanced-learn matplotlib seaborn

Cómo correr:
  python 07_churn_model.py
"""

import sqlite3
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from config import DB_PATH

warnings.filterwarnings("ignore")

# ── Estilo global ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLOR_POS  = "#E8604C"   # churned
COLOR_NEG  = "#4C9BE8"   # activo
COLOR_XGB  = "#E8604C"
COLOR_LR   = "#4C9BE8"

SEGMENT_COLORS = {
    "💎 Cliente Premium"    : "#4C9BE8",
    "🌟 Cliente Leal"       : "#4CAF50",
    "🏷️  Cazador de Ofertas": "#FF9800",
    "⚠️  Cliente en Riesgo" : "#E8604C",
}

# ── Features del modelo ───────────────────────────────────────────────────────
# Excluimos discount_usage_rate y price_sensitivity_index juntos (r=0.96, colineales)
# Nos quedamos con price_sensitivity_index por ser más informativo
FEATURES = [
    "account_age_months",
    "avg_order_value",
    "total_orders",
    "days_since_last_purchase",
    "price_sensitivity_index",
    "browsing_frequency_per_week",
    "cart_abandonment_rate",
    "return_rate",
    "customer_support_tickets",
    "product_review_score_avg",
    "engagement_score",
    "satisfaction_score",
    "loyalty_member",           # 1/0 codificado abajo
]

FEATURE_LABELS = {
    "account_age_months"         : "Antigüedad cuenta",
    "avg_order_value"            : "Valor promedio orden",
    "total_orders"               : "Total órdenes",
    "days_since_last_purchase"   : "Días sin comprar",
    "price_sensitivity_index"    : "Sensibilidad al precio",
    "browsing_frequency_per_week": "Frec. navegación",
    "cart_abandonment_rate"      : "Abandono carrito",
    "return_rate"                : "Tasa devolución",
    "customer_support_tickets"   : "Tickets soporte",
    "product_review_score_avg"   : "Score review",
    "engagement_score"           : "Engagement",
    "satisfaction_score"         : "Satisfacción",
    "loyalty_member"             : "Loyalty member",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def insight(text: str) -> None:
    border = "─" * 62
    wrapped = textwrap.fill(text.strip(), width=60)
    print(f"\n  ┌{border}┐")
    for line in wrapped.splitlines():
        print(f"  │  {line:<60}│")
    print(f"  └{border}┘")

def section_header(num: int, title: str) -> None:
    print()
    print("═" * 66)
    print(f"  PASO {num}: {title}")
    print("═" * 66)


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — Cargar datos y feature engineering
# ══════════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT
            fc.customer_id,
            fc.account_age_months,
            fc.avg_order_value,
            fc.total_orders,
            fc.days_since_last_purchase,
            fc.price_sensitivity_index,
            fc.browsing_frequency_per_week,
            fc.cart_abandonment_rate,
            fc.return_rate,
            fc.customer_support_tickets,
            fc.product_review_score_avg,
            fc.engagement_score,
            fc.satisfaction_score,
            fc.churned,
            CASE ls.status_name WHEN 'Yes' THEN 1 ELSE 0 END AS loyalty_member,
            cs.segment_name
        FROM fact_customers fc
        JOIN dim_loyalty_status ls ON fc.loyalty_status_id = ls.loyalty_status_id
        LEFT JOIN customer_segments cs ON fc.customer_id = cs.customer_id
    """, conn)
    conn.close()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Split + SMOTE
# ══════════════════════════════════════════════════════════════════════════════
def preparar_datos(df: pd.DataFrame):
    section_header(2, "Split train/test + balanceo con SMOTE")

    X = df[FEATURES].copy()
    y = df["churned"].astype(int)

    print(f"\n  Distribución original:")
    print(f"    Activos  (0): {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
    print(f"    Churned  (1): {(y==1).sum():,}  ({(y==1).mean()*100:.1f}%)")

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE solo en train para balancear la clase minoritaria
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"\n  Train tras SMOTE:")
    print(f"    Activos  (0): {(y_train_sm==0).sum():,}")
    print(f"    Churned  (1): {(y_train_sm==1).sum():,}")
    print(f"\n  Test (sin modificar):")
    print(f"    Activos  (0): {(y_test==0).sum():,}")
    print(f"    Churned  (1): {(y_test==1).sum():,}")

    insight(
        "SMOTE (Synthetic Minority Oversampling Technique) genera ejemplos "
        "sintéticos de la clase minoritaria (churned=1) solo en el set de "
        "entrenamiento. El test permanece intacto para evaluar con datos reales."
    )

    # Escalar para Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sm, X_test, y_train_sm, y_test, X_train_sc, X_test_sc, scaler


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Logistic Regression (baseline)
# ══════════════════════════════════════════════════════════════════════════════
def entrenar_lr(X_train_sc, X_test_sc, y_train, y_test):
    section_header(3, "Entrenando Logistic Regression (baseline)")

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)

    y_pred  = lr.predict(X_test_sc)
    y_proba = lr.predict_proba(X_test_sc)[:, 1]

    metricas = calcular_metricas(y_test, y_pred, y_proba, "Logistic Regression")
    return lr, y_pred, y_proba, metricas


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — XGBoost (modelo final)
# ══════════════════════════════════════════════════════════════════════════════
def entrenar_xgb(X_train, X_test, y_train, y_test):
    section_header(4, "Entrenando XGBoost (modelo final)")

    xgb = XGBClassifier(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        use_label_encoder=False,
        eval_metric     = "logloss",
        random_state    = 42,
        verbosity       = 0,
    )
    xgb.fit(X_train, y_train)

    y_pred  = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]

    metricas = calcular_metricas(y_test, y_pred, y_proba, "XGBoost")
    return xgb, y_pred, y_proba, metricas


# ── Métricas ──────────────────────────────────────────────────────────────────
def calcular_metricas(y_test, y_pred, y_proba, nombre: str) -> dict:
    m = {
        "modelo"    : nombre,
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred),
        "recall"    : recall_score(y_test, y_pred),
        "f1"        : f1_score(y_test, y_pred),
        "roc_auc"   : roc_auc_score(y_test, y_proba),
    }
    print(f"\n  {nombre}:")
    print(f"    Accuracy  : {m['accuracy']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1 Score  : {m['f1']:.4f}")
    print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Activo", "Churned"]))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Comparar modelos
# ══════════════════════════════════════════════════════════════════════════════
def comparar_modelos(m_lr: dict, m_xgb: dict) -> None:
    section_header(5, "Comparación de modelos")

    df_comp = pd.DataFrame([m_lr, m_xgb]).set_index("modelo")
    print(f"\n  {'Métrica':<12} {'Log. Reg.':>12} {'XGBoost':>12} {'Ganador':>12}")
    print("  " + "─" * 50)
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lr_val  = df_comp.loc["Logistic Regression", col]
        xgb_val = df_comp.loc["XGBoost", col]
        winner  = "XGBoost" if xgb_val >= lr_val else "Log. Reg."
        print(f"  {col:<12} {lr_val:>12.4f} {xgb_val:>12.4f} {winner:>12}")

    mejor_auc = "XGBoost" if m_xgb["roc_auc"] >= m_lr["roc_auc"] else "Logistic Regression"
    insight(
        f"El modelo con mejor ROC-AUC es {mejor_auc} "
        f"(AUC={max(m_lr['roc_auc'], m_xgb['roc_auc']):.4f}). "
        f"En churn prediction el Recall es crítico: queremos minimizar "
        f"los falsos negativos (clientes que se van sin que lo detectemos)."
    )
    insight(
        "El ROC-AUC mide la capacidad del modelo de distinguir entre "
        "churned y activos independientemente del umbral de decisión. "
        "Un AUC=1.0 es perfecto; AUC=0.5 es equivalente a adivinar al azar."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — Visualizaciones
# ══════════════════════════════════════════════════════════════════════════════
def visualizar(df: pd.DataFrame,
               lr, xgb_model,
               X_test_lr, X_test_xgb,
               y_test,
               y_proba_lr, y_proba_xgb,
               y_pred_lr, y_pred_xgb) -> None:
    section_header(6, "Visualizaciones del modelo")

    # ── Figura 6A: ROC Curves ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Paso 6A — Curvas ROC y Matrices de Confusión",
                 fontsize=13, fontweight="bold")

    for ax, nombre, y_proba, y_pred, color in [
        (axes[0], "Logistic Regression", y_proba_lr,  y_pred_lr,  COLOR_LR),
        (axes[1], "XGBoost",             y_proba_xgb, y_pred_xgb, COLOR_XGB),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"AUC = {auc:.4f}")
        ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Random (0.50)")
        ax.fill_between(fpr, tpr, alpha=0.1, color=color)
        ax.set_title(f"ROC — {nombre}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # ── Figura 6B: Confusion Matrices ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Paso 6B — Matrices de Confusión",
                 fontsize=13, fontweight="bold")

    for ax, nombre, y_pred, color in [
        (axes[0], "Logistic Regression", y_pred_lr,  COLOR_LR),
        (axes[1], "XGBoost",             y_pred_xgb, COLOR_XGB),
    ]:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(color, as_cmap=True),
                    ax=ax, linewidths=0.5,
                    xticklabels=["Activo", "Churned"],
                    yticklabels=["Activo", "Churned"])
        ax.set_title(nombre)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.show()

    # ── Figura 6C: Feature Importance XGBoost ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Paso 6C — Importancia de variables",
                 fontsize=13, fontweight="bold")

    # XGBoost feature importance
    importances_xgb = pd.Series(
        xgb_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)
    importances_xgb.index = [FEATURE_LABELS[f] for f in importances_xgb.index]

    colors_imp = [COLOR_XGB if v == importances_xgb.max() else "#4C9BE8"
                  for v in importances_xgb.values]
    importances_xgb.plot(kind="barh", ax=axes[0], color=colors_imp, edgecolor="white")
    axes[0].set_title("XGBoost — Feature Importance")
    axes[0].set_xlabel("Importancia (F-score)")
    for i, v in enumerate(importances_xgb.values):
        axes[0].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)

    # Logistic Regression coeficientes
    coefs = pd.Series(
        np.abs(lr.coef_[0]), index=FEATURES
    ).sort_values(ascending=True)
    coefs.index = [FEATURE_LABELS[f] for f in coefs.index]
    colors_coef = [COLOR_LR if v == coefs.max() else "#6A9E72"
                   for v in coefs.values]
    coefs.plot(kind="barh", ax=axes[1], color=colors_coef, edgecolor="white")
    axes[1].set_title("Logistic Regression — |Coeficientes|")
    axes[1].set_xlabel("|Coeficiente| (importancia)")
    for i, v in enumerate(coefs.values):
        axes[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    # ── Figura 6D: Distribución del score de riesgo por segmento ─────────────
    # Calcular score XGBoost para todos los clientes
    X_all    = df[FEATURES].copy()
    scaler   = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all)   # solo para visualización
    scores_all = xgb_model.predict_proba(X_all)[:, 1]
    df["churn_score"] = scores_all

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Paso 6D — Score de riesgo de churn por segmento",
                 fontsize=13, fontweight="bold")

    seg_order  = list(SEGMENT_COLORS.keys())
    seg_colors = list(SEGMENT_COLORS.values())

    # Boxplot por segmento
    df_seg = df[df["segment_name"].notna()]
    sns.boxplot(data=df_seg, x="segment_name", y="churn_score",
                order=seg_order, palette=SEGMENT_COLORS,
                ax=axes[0], width=0.55, fliersize=2)
    axes[0].set_title("Distribución del score de riesgo por segmento")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Probabilidad de churn")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1.2,
                    label="Umbral 0.50")
    axes[0].legend()

    # % clientes por encima del umbral 0.5 por segmento
    riesgo_pct = (
        df_seg.groupby("segment_name")["churn_score"]
        .apply(lambda x: (x >= 0.5).mean() * 100)
        .reindex(seg_order)
    )
    bars = axes[1].bar(seg_order,  riesgo_pct,
                       color=seg_colors, edgecolor="white", width=0.6)
    axes[1].set_title("% clientes con score ≥ 0.50 por segmento")
    axes[1].set_ylabel("% en riesgo alto")
    axes[1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, riesgo_pct):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    # ── Figura 6E: histograma de scores — real vs predicho ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Paso 6E — Distribución del score de riesgo (XGBoost)",
                 fontsize=13, fontweight="bold")

    # En test set, separado por clase real
    df_test = df.iloc[y_test.index] if hasattr(y_test, "index") else df.copy()
    score_activo  = y_proba_xgb[y_test.values == 0]
    score_churned = y_proba_xgb[y_test.values == 1]

    axes[0].hist(score_activo,  bins=40, color=COLOR_NEG, alpha=0.6,
                 label="Real: Activo",  density=True)
    axes[0].hist(score_churned, bins=40, color=COLOR_POS, alpha=0.6,
                 label="Real: Churned", density=True)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5,
                    label="Umbral 0.50")
    axes[0].set_title("Score por clase real (test set)")
    axes[0].set_xlabel("Probabilidad de churn predicha")
    axes[0].set_ylabel("Densidad")
    axes[0].legend()

    # Score global todos los clientes
    axes[1].hist(df["churn_score"], bins=50, color="#6A9E72",
                 edgecolor="white", alpha=0.85)
    axes[1].axvline(0.5, color=COLOR_POS, linestyle="--", linewidth=1.5,
                    label=f"Umbral 0.50 → {(df['churn_score']>=0.5).sum():,} clientes")
    axes[1].set_title("Distribución del score — todos los clientes")
    axes[1].set_xlabel("Probabilidad de churn")
    axes[1].set_ylabel("Número de clientes")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PASO 7 — Persistir score en la DB
# ══════════════════════════════════════════════════════════════════════════════
def persistir_scores(df: pd.DataFrame, xgb_model) -> None:
    section_header(7, "Guardando scores de riesgo en la DB")

    X_all      = df[FEATURES].copy()
    scores     = xgb_model.predict_proba(X_all)[:, 1]
    predicciones = xgb_model.predict(X_all)

    df_scores = pd.DataFrame({
        "customer_id"     : df["customer_id"].values,
        "churn_score"     : scores.round(6),
        "churn_predicted" : predicciones,         # 1 = en riesgo, 0 = activo
        "risk_tier"       : pd.cut(
            scores,
            bins   = [0, 0.25, 0.50, 0.75, 1.01],
            labels = ["Bajo", "Medio", "Alto", "Crítico"]
        ).astype(str),
    })

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            customer_id      TEXT PRIMARY KEY REFERENCES fact_customers(customer_id),
            churn_score      REAL    NOT NULL,
            churn_predicted  INTEGER NOT NULL,
            risk_tier        TEXT    NOT NULL
        )
    """)
    conn.execute("DELETE FROM churn_scores")
    conn.commit()

    df_scores.to_sql("churn_scores", conn, if_exists="append", index=False)

    # Resumen
    dist = conn.execute("""
        SELECT risk_tier, COUNT(*) as clientes,
               ROUND(AVG(churn_score)*100, 1) as avg_score_pct
        FROM churn_scores
        GROUP BY risk_tier
        ORDER BY avg_score_pct DESC
    """).fetchall()

    print(f"\n  {'Tier de riesgo':<12} {'Clientes':>10} {'Score prom.':>12}")
    print("  " + "─" * 36)
    for tier, n, avg in dist:
        print(f"  {tier:<12} {n:>10,} {avg:>11.1f}%")

    conn.close()

    insight(
        f"Score guardado en tabla 'churn_scores'. Cada cliente tiene: "
        f"churn_score (0.0–1.0), churn_predicted (0/1) y risk_tier "
        f"(Bajo / Medio / Alto / Crítico). Listo para cruzar con segmentos "
        f"en Power BI o para campañas de retención."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=== 07 — Modelo Predictivo de Churn ===")
    log("Cierra cada ventana de gráfica para avanzar.\n")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB. Corre primero los scripts 01–06."
        )

    # Paso 1
    section_header(1, "Cargando datos + feature engineering")
    df = load_data()
    log(f"  Clientes: {len(df):,} | Features: {len(FEATURES)}")
    log(f"  Churn rate: {df['churned'].mean()*100:.1f}%")

    # Paso 2
    X_train, X_test, y_train, y_test, \
    X_train_sc, X_test_sc, scaler = preparar_datos(df)

    # Paso 3 — Logistic Regression
    lr, y_pred_lr, y_proba_lr, m_lr = entrenar_lr(
        X_train_sc, X_test_sc, y_train, y_test
    )

    # Paso 4 — XGBoost
    xgb_model, y_pred_xgb, y_proba_xgb, m_xgb = entrenar_xgb(
        X_train, X_test, y_train, y_test
    )

    # Paso 5 — Comparación
    comparar_modelos(m_lr, m_xgb)

    # Paso 6 — Visualizaciones
    visualizar(
        df, lr, xgb_model,
        X_test_sc, X_test,
        y_test,
        y_proba_lr, y_proba_xgb,
        y_pred_lr, y_pred_xgb,
    )

    # Paso 7 — Persistir
    persistir_scores(df, xgb_model)

    log("✅  Modelo de churn completado. Tabla 'churn_scores' lista en la DB.")


if __name__ == "__main__":
    main()
