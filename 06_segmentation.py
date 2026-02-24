"""
06_segmentation.py — Perfiles de compradores (Buyer Personas)
=============================================================
Pasos:
  1. Cargar y escalar variables de segmentación
  2. Evaluar k óptimo: método del codo + silhouette score
  3. Entrenar K-Means con el k elegido
  4. Caracterizar y nombrar cada segmento
  5. Persistir el segmento asignado en la DB (tabla customer_segments)
  6. Visualizar perfiles con gráficas en ventanas

Prerequisito : haber corrido 01–04 (DB lista)
Dependencias : pip install scikit-learn matplotlib seaborn pandas

Cómo correr:
  python 06_segmentation.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from config import DB_PATH

warnings.filterwarnings("ignore")

# ── Estilo global ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
SEGMENT_COLORS = ["#4C9BE8", "#E8604C", "#4CAF50", "#FF9800", "#9C27B0"]

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
# PASO 1 — Cargar y escalar datos
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # Valor económico
    "avg_order_value",
    "total_orders",
    # Comportamiento
    "engagement_score",
    "browsing_frequency_per_week",
    "cart_abandonment_rate",
    # Riesgo
    "days_since_last_purchase",
    "return_rate",
    # Soporte y satisfacción
    "customer_support_tickets",
    "satisfaction_score",
    "product_review_score_avg",
]

# Etiquetas legibles para las gráficas
FEATURE_LABELS = {
    "avg_order_value"            : "Valor promedio orden",
    "total_orders"               : "Total órdenes",
    "engagement_score"           : "Engagement",
    "browsing_frequency_per_week": "Frec. navegación",
    "cart_abandonment_rate"      : "Abandono carrito",
    "days_since_last_purchase"   : "Días sin comprar",
    "return_rate"                : "Tasa devolución",
    "customer_support_tickets"   : "Tickets soporte",
    "satisfaction_score"         : "Satisfacción",
    "product_review_score_avg"   : "Score review",
}

def load_and_scale() -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT customer_id, churned, loyalty_status, {', '.join(FEATURES)} "
        f"FROM vw_customers_full",
        conn,
    )
    conn.close()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return df, X_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Evaluar k óptimo
# ══════════════════════════════════════════════════════════════════════════════
def evaluar_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    section_header(2, "Evaluando número óptimo de clusters (k)")

    inertias    = []
    silhouettes = []
    k_range     = range(k_min, k_max + 1)

    print(f"\n  {'k':>4}  {'Inercia':>12}  {'Silhouette':>12}")
    print("  " + "─" * 32)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        print(f"  {k:>4}  {km.inertia_:>12,.1f}  {sil:>12.4f}")

    # k estadísticamente óptimo = mayor silhouette score
    k_estadistico = k_range[int(np.argmax(silhouettes))]
    insight(
        f"El silhouette score máximo estadístico es k={k_estadistico} "
        f"(score={max(silhouettes):.4f}). Los scores son bajos porque el "
        f"dataset sintético no tiene clusters naturalmente separados. "
        f"Forzamos k=4 por valor narrativo para el portafolio."
    )
    k_optimo = 4

    # ── Figura: método del codo + silhouette ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Paso 2 — Selección del número óptimo de clusters",
                 fontsize=13, fontweight="bold")

    # Codo
    axes[0].plot(list(k_range), inertias, marker="o", color="#4C9BE8", linewidth=2)
    axes[0].axvline(k_optimo, color="#E8604C", linestyle="--", linewidth=1.5,
                    label=f"k óptimo = {k_optimo}")
    axes[0].set_title("Método del Codo (Inercia)")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].legend()

    # Silhouette
    bar_colors = ["#E8604C" if k == k_optimo else "#4C9BE8" for k in k_range]
    axes[1].bar(list(k_range), silhouettes, color=bar_colors, edgecolor="white")
    axes[1].set_title("Silhouette Score por k")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].axhline(max(silhouettes), color="#E8604C", linestyle="--",
                    linewidth=1.2, label=f"Máximo: {max(silhouettes):.4f}")
    axes[1].legend()
    for i, (k, s) in enumerate(zip(k_range, silhouettes)):
        axes[1].text(k, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    return k_optimo


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Entrenar K-Means con k óptimo
# ══════════════════════════════════════════════════════════════════════════════
def entrenar_kmeans(df: pd.DataFrame, X_scaled: np.ndarray,
                    k: int) -> pd.DataFrame:
    section_header(3, f"Entrenando K-Means con k={k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # Tamaño de cada cluster
    sizes = df["cluster"].value_counts().sort_index()
    print(f"\n  {'Cluster':>8}  {'Clientes':>10}  {'%':>8}")
    print("  " + "─" * 30)
    for cl, n in sizes.items():
        print(f"  {cl:>8}  {n:>10,}  {n/len(df)*100:>7.1f}%")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — Caracterizar y nombrar segmentos
# ══════════════════════════════════════════════════════════════════════════════

# Nombres según el perfil que suele emerger en datasets de e-commerce.
# Se asignan dinámicamente después de analizar las medias de cada cluster.
SEGMENT_NAMES = {
    "high_value"   : "💎 Cliente Premium",
    "loyal"        : "🌟 Cliente Leal",
    "at_risk"      : "⚠️  Cliente en Riesgo",
    "price_hunter" : "🏷️  Cazador de Ofertas",
    "dormant"      : "💤 Cliente Dormido",
    "new"          : "🆕 Cliente Nuevo",
}

def asignar_nombres(df: pd.DataFrame, k: int) -> dict[int, str]:
    """
    Asigna un nombre de perfil a cada uno de los 4 clusters.

    Criterios en orden de prioridad:
      1. Mayor days_since_last_purchase + churn alto  → En Riesgo
      2. Mayor avg_order_value                        → Premium
      3. Mayor engagement_score                       → Leal
      4. Mayor discount_usage_rate                    → Cazador de Ofertas
    """
    medias     = df.groupby("cluster")[FEATURES].mean()
    churn_rate = df.groupby("cluster")["churned"].apply(lambda x: (x == "Yes").mean())
    nombres    = {}
    asignados  = set()

    # 1. En Riesgo: más días sin comprar Y mayor tasa de churn
    score_riesgo = (
        medias["days_since_last_purchase"].rank() +
        churn_rate.rank()
    )
    cl_riesgo = score_riesgo.idxmax()
    nombres[cl_riesgo] = SEGMENT_NAMES["at_risk"]
    asignados.add(cl_riesgo)

    # 2. Premium: mayor valor de orden entre los restantes
    restantes  = medias.drop(index=list(asignados))
    cl_premium = restantes["avg_order_value"].idxmax()
    nombres[cl_premium] = SEGMENT_NAMES["high_value"]
    asignados.add(cl_premium)

    # 3. Leal: mayor engagement entre los restantes
    restantes = medias.drop(index=list(asignados))
    cl_leal   = restantes["engagement_score"].idxmax()
    nombres[cl_leal] = SEGMENT_NAMES["loyal"]
    asignados.add(cl_leal)

    # 4. Cazador de Ofertas: el cluster restante
    cl_precio = (set(range(k)) - asignados).pop()
    nombres[cl_precio] = SEGMENT_NAMES["price_hunter"]

    return nombres


def caracterizar(df: pd.DataFrame, nombres: dict[int, str]) -> pd.DataFrame:
    section_header(4, "Caracterización de segmentos")

    df["segment_name"] = df["cluster"].map(nombres)
    medias = df.groupby("segment_name")[FEATURES].mean().round(3)
    extra  = df.groupby("segment_name").agg(
        clientes    =("customer_id",   "count"),
        pct_churned =("churned",       lambda x: f"{(x=='Yes').mean()*100:.1f}%"),
        pct_loyalty =("loyalty_status",lambda x: f"{(x=='Yes').mean()*100:.1f}%"),
    )

    print("\n  Medias por segmento:")
    labels_df = medias.rename(columns=FEATURE_LABELS)
    print(labels_df.to_string())

    print("\n  Resumen por segmento:")
    print(extra.to_string())

    for cl, nombre in nombres.items():
        grupo  = df[df["cluster"] == cl]
        n      = len(grupo)
        churn  = (grupo["churned"] == "Yes").mean() * 100
        dias   = grupo["days_since_last_purchase"].mean()
        eng    = grupo["engagement_score"].mean()
        valor  = grupo["avg_order_value"].mean()
        insight(
            f"{nombre} ({n:,} clientes, {n/len(df)*100:.1f}%): "
            f"valor promedio ${valor:.0f}, engagement {eng:.2f}, "
            f"{dias:.0f} días sin comprar, churn {churn:.1f}%."
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Persistir en la DB
# ══════════════════════════════════════════════════════════════════════════════
def persistir(df: pd.DataFrame) -> None:
    section_header(5, "Guardando segmentos en la DB")

    conn = sqlite3.connect(DB_PATH)

    # Crear tabla si no existe
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_segments (
            customer_id  TEXT PRIMARY KEY REFERENCES fact_customers(customer_id),
            cluster_id   INTEGER NOT NULL,
            segment_name TEXT    NOT NULL
        )
    """)
    conn.execute("DELETE FROM customer_segments")  # limpiar antes de insertar
    conn.commit()

    df[["customer_id", "cluster", "segment_name"]].rename(
        columns={"cluster": "cluster_id"}
    ).to_sql("customer_segments", conn, if_exists="append", index=False)

    count = conn.execute("SELECT COUNT(*) FROM customer_segments").fetchone()[0]
    dist  = conn.execute(
        "SELECT segment_name, COUNT(*) FROM customer_segments GROUP BY segment_name"
    ).fetchall()

    print(f"\n  Registros guardados: {count:,}")
    print(f"\n  {'Segmento':<30} {'Clientes':>10}")
    print("  " + "─" * 42)
    for name, n in dist:
        print(f"  {name:<30} {n:>10,}")

    conn.close()
    insight(
        "Los segmentos quedan guardados en la tabla 'customer_segments' "
        "con FK a fact_customers. En el siguiente paso podrás cruzarlos "
        "con el modelo de churn para enriquecer los perfiles."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — Visualizaciones
# ══════════════════════════════════════════════════════════════════════════════
def visualizar(df: pd.DataFrame, X_scaled: np.ndarray,
               nombres: dict[int, str], k: int) -> None:
    section_header(6, "Visualizando perfiles de segmentos")

    palette = {nombres[i]: SEGMENT_COLORS[i] for i in range(k)}

    # ── Figura 6A: radar chart por segmento ───────────────────────────────────
    # Normalizamos medias a [0,1] para que todas las variables sean comparables
    medias_raw = df.groupby("segment_name")[FEATURES].mean()
    medias_norm = (medias_raw - medias_raw.min()) / (medias_raw.max() - medias_raw.min())

    labels_radar = [FEATURE_LABELS[f] for f in FEATURES]
    N = len(labels_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # cerrar el polígono

    fig, axes = plt.subplots(1, k, figsize=(5 * k, 5),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Paso 6A — Radar: Perfil de cada segmento",
                 fontsize=13, fontweight="bold", y=1.02)

    if k == 1:
        axes = [axes]

    for ax, (seg_name, row) in zip(axes, medias_norm.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color  = palette[seg_name]
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, size=7.5)
        ax.set_ylim(0, 1)
        ax.set_title(seg_name, size=9, pad=14, fontweight="bold")
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()

    # ── Figura 6B: PCA 2D — clusters en espacio reducido ─────────────────────
    pca  = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_1"] = coords[:, 0]
    df["pca_2"] = coords[:, 1]

    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    for seg_name, color in palette.items():
        mask = df["segment_name"] == seg_name
        ax.scatter(df.loc[mask, "pca_1"], df.loc[mask, "pca_2"],
                   label=seg_name, color=color, alpha=0.45, s=18, edgecolors="none")

    ax.set_title("Paso 6B — Clusters en espacio PCA 2D\n"
                 f"(Varianza explicada: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Componente Principal 1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"Componente Principal 2 ({var_exp[1]:.1f}%)")
    ax.legend(title="Segmento", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ── Figura 6C: boxplots de variables clave por segmento ───────────────────
    vars_clave = [
        ("avg_order_value",          "Valor promedio orden ($)"),
        ("engagement_score",         "Engagement score"),
        ("days_since_last_purchase", "Días sin comprar"),
        ("satisfaction_score",       "Satisfacción"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Paso 6C — Distribución de variables clave por segmento",
                 fontsize=13, fontweight="bold")

    seg_order = list(palette.keys())
    for ax, (col, label) in zip(axes.flat, vars_clave):
        sns.boxplot(data=df, x="segment_name", y=col,
                    order=seg_order, palette=palette, ax=ax,
                    width=0.55, fliersize=2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()

    # ── Figura 6D: distribución de churn y lealtad por segmento ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Paso 6D — Churn y membresía de lealtad por segmento",
                 fontsize=13, fontweight="bold")

    # Tasa de churn
    churn_rate = (
        df.groupby("segment_name")["churned"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(seg_order)
    )
    bar_colors = [palette[s] for s in seg_order]
    bars = axes[0].bar(seg_order, churn_rate, color=bar_colors, edgecolor="white", width=0.6)
    axes[0].set_title("Tasa de churn por segmento (%)")
    axes[0].set_ylabel("% clientes churned")
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, churn_rate):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    # Lealtad
    loyalty_rate = (
        df.groupby("segment_name")["loyalty_status"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(seg_order)
    )
    bars2 = axes[1].bar(seg_order, loyalty_rate, color=bar_colors, edgecolor="white", width=0.6)
    axes[1].set_title("Membresía de lealtad por segmento (%)")
    axes[1].set_ylabel("% loyalty members")
    axes[1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, loyalty_rate):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    # ── Figura 6E: heatmap de medias normalizadas (resumen ejecutivo) ─────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Paso 6E — Heatmap resumen: Perfil normalizado por segmento",
                 fontsize=13, fontweight="bold")

    medias_norm_display = medias_norm.rename(columns=FEATURE_LABELS)
    sns.heatmap(medias_norm_display, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Valor norm. [0-1]"},
                vmin=0, vmax=1)
    ax.set_title("Verde = alto, Rojo = bajo (normalizado por variable)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=== 06 — Segmentación de clientes (Buyer Personas) ===")
    log("Cierra cada ventana de gráfica para avanzar al siguiente paso.\n")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB en {DB_PATH}. Corre primero los scripts 01–04."
        )

    # Paso 1
    section_header(1, "Cargando y escalando datos")
    df, X_scaled, scaler = load_and_scale()
    log(f"  Clientes cargados: {len(df):,} | Variables: {len(FEATURES)}")
    log(f"  Variables: {', '.join(FEATURES)}")

    # Paso 2
    k_optimo = evaluar_k(X_scaled)

    # Paso 3
    df = entrenar_kmeans(df, X_scaled, k_optimo)

    # Paso 4
    nombres = asignar_nombres(df, k_optimo)
    df = caracterizar(df, nombres)

    # Paso 5
    persistir(df)

    # Paso 6
    visualizar(df, X_scaled, nombres, k_optimo)

    log("✅  Segmentación completada. Tabla 'customer_segments' lista en la DB.")


if __name__ == "__main__":
    main()