"""
05_eda.py — Análisis Exploratorio de Datos (EDA)
=================================================
Secciones:
  1. Distribuciones y estadísticas descriptivas
  2. Correlaciones entre variables
  3. Análisis de churn
  4. Análisis temporal de órdenes

Cada sección imprime insights en consola y abre una ventana con gráficas.
Cierra cada ventana para avanzar a la siguiente sección.

Prerequisito: haber corrido 01 04 (DB lista).
Dependencias:  pip install matplotlib seaborn pandas

Cómo correr:
  python 05_eda.py
"""

import sqlite3
import warnings
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from config import DB_PATH

warnings.filterwarnings("ignore")

# ── Estilo global ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE_CHURN  = {"No": "#4C9BE8", "Yes": "#E8604C"}
PALETTE_STATUS = {"Delivered": "#4CAF50", "Cancelled": "#FF9800", "Returned": "#E8604C"}
COL_ACCENT     = "#4C9BE8"

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def insight(text: str) -> None:
    """Imprime un bloque de insight formateado en consola."""
    border = "─" * 62
    wrapped = textwrap.fill(text.strip(), width=60)
    print(f"\n  ┌{border}┐")
    for line in wrapped.splitlines():
        print(f"  │  {line:<60}│")
    print(f"  └{border}┘")

def section_header(num: int, title: str) -> None:
    print()
    print("═" * 66)
    print(f"  SECCIÓN {num}: {title}")
    print("═" * 66)

# ── Carga de datos ────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(DB_PATH)
    df_c = pd.read_sql("SELECT * FROM vw_customers_full", conn)
    df_o = pd.read_sql("SELECT * FROM vw_orders_full", conn)
    conn.close()

    # Tipos correctos
    df_o["order_date"] = pd.to_datetime(df_o["order_date"])
    df_o["month_num"]  = df_o["order_date"].dt.month

    return df_c, df_o


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Distribuciones y estadísticas descriptivas
# ══════════════════════════════════════════════════════════════════════════════
def seccion_1(df_c: pd.DataFrame, df_o: pd.DataFrame) -> None:
    section_header(1, "Distribuciones y estadísticas descriptivas")

    # ── Estadísticas en consola ───────────────────────────────────────────────
    num_cols = [
        "account_age_months", "avg_order_value", "total_orders",
        "days_since_last_purchase", "engagement_score", "satisfaction_score",
    ]
    stats = df_c[num_cols].describe().round(2)
    print("\n  Clientes — estadísticas descriptivas:")
    print(stats.to_string())

    print("\n  Órdenes — estadísticas descriptivas:")
    print(df_o[["order_value", "delivery_time_days", "customer_rating"]].describe().round(2).to_string())

    # Insights
    insight(
        f"El valor promedio de orden de los clientes es ${df_c['avg_order_value'].mean():.2f}, "
        f"con una mediana de ${df_c['avg_order_value'].median():.2f}. La diferencia entre ambas "
        f"indica una distribución con cola derecha: hay clientes de alto valor que jalan el promedio hacia arriba."
    )
    insight(
        f"La antigüedad promedio de cuenta es {df_c['account_age_months'].mean():.1f} meses "
        f"({df_c['account_age_months'].mean()/12:.1f} años). El 25% de los clientes tiene menos de "
        f"{df_c['account_age_months'].quantile(0.25):.0f} meses — una base relativamente joven."
    )
    insight(
        f"El rating promedio de órdenes es {df_o['customer_rating'].mean():.2f}/5. "
        f"El tiempo de entrega promedio es {df_o['delivery_time_days'].mean():.1f} días, "
        f"con máximo de {df_o['delivery_time_days'].max()} días."
    )

    # ── Figura 1A: distribuciones de clientes ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Sección 1A — Distribuciones: Variables de Clientes", fontsize=14, fontweight="bold", y=1.01)

    plot_data = [
        ("avg_order_value",          "Valor promedio de orden ($)"),
        ("total_orders",             "Total de órdenes"),
        ("account_age_months",       "Antigüedad de cuenta (meses)"),
        ("days_since_last_purchase", "Días desde última compra"),
        ("engagement_score",         "Engagement score"),
        ("satisfaction_score",       "Satisfaction score"),
    ]
    for ax, (col, label) in zip(axes.flat, plot_data):
        sns.histplot(df_c[col], kde=True, color=COL_ACCENT, ax=ax, bins=30)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        mean_val = df_c[col].mean()
        ax.axvline(mean_val, color="#E8604C", linestyle="--", linewidth=1.2, label=f"Media: {mean_val:.1f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # ── Figura 1B: distribuciones de órdenes ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sección 1B — Distribuciones: Variables de Órdenes", fontsize=14, fontweight="bold")

    order_plots = [
        ("order_value",         "Valor de orden ($)"),
        ("delivery_time_days",  "Días de entrega"),
        ("customer_rating",     "Rating del cliente"),
    ]
    for ax, (col, label) in zip(axes, order_plots):
        sns.histplot(df_o[col], kde=True, color="#6A9E72", ax=ax, bins=25)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        mean_val = df_o[col].mean()
        ax.axvline(mean_val, color="#E8604C", linestyle="--", linewidth=1.2, label=f"Media: {mean_val:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # ── Figura 1C: variables categóricas de órdenes ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sección 1C — Variables Categóricas: Órdenes", fontsize=14, fontweight="bold")

    cat_plots = [
        ("product_category", "Categoría de producto"),
        ("payment_method",   "Método de pago"),
        ("order_status",     "Estado de orden"),
    ]
    colors_list = [
        sns.color_palette("muted", 6),
        sns.color_palette("muted", 4),
        [PALETTE_STATUS["Delivered"], PALETTE_STATUS["Cancelled"], PALETTE_STATUS["Returned"]],
    ]
    for ax, (col, label), colors in zip(axes, cat_plots, colors_list):
        counts = df_o[col].value_counts()
        bars = ax.bar(counts.index, counts.values, color=colors)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Correlaciones
# ══════════════════════════════════════════════════════════════════════════════
def seccion_2(df_c: pd.DataFrame, df_o: pd.DataFrame) -> None:
    section_header(2, "Correlaciones entre variables")

    num_cols_c = [
        "account_age_months", "avg_order_value", "total_orders",
        "days_since_last_purchase", "discount_usage_rate", "return_rate",
        "customer_support_tickets", "product_review_score_avg",
        "browsing_frequency_per_week", "cart_abandonment_rate",
        "engagement_score", "satisfaction_score", "price_sensitivity_index",
    ]
    corr_c = df_c[num_cols_c].corr()

    # Top correlaciones absolutas
    corr_pairs = (
        corr_c.where(np.triu(np.ones(corr_c.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["var1", "var2", "corr"]
    corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
    top_corr = corr_pairs.nlargest(5, "abs_corr")

    print("\n  Top 5 correlaciones más fuertes (clientes):")
    for _, row in top_corr.iterrows():
        print(f"    {row['var1']:35s} ↔ {row['var2']:35s}  r = {row['corr']:+.3f}")

    insight(
        "El engagement_score y satisfaction_score muestran la correlación más "
        "fuerte entre las métricas compuestas. Esto es esperado ya que ambas "
        "capturan bienestar del cliente desde ángulos distintos pero relacionados."
    )
    insight(
        "La tasa de descuento (discount_usage_rate) y la sensibilidad al precio "
        "(price_sensitivity_index) tienden a correlacionarse — los clientes más "
        "sensibles al precio buscan activamente descuentos."
    )

    # ── Figura 2A: heatmap clientes ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_c, dtype=bool))
    sns.heatmap(
        corr_c, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, vmin=-1, vmax=1,
    )
    ax.set_title("Sección 2A — Mapa de Correlación: Variables de Clientes",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()

    # ── Figura 2B: correlación variables de órdenes ───────────────────────────
    num_cols_o = ["order_value", "delivery_time_days", "customer_rating", "customer_age"]
    corr_o = df_o[num_cols_o].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sección 2B — Correlaciones: Órdenes", fontsize=13, fontweight="bold")

    sns.heatmap(corr_o, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title("Correlación entre variables de órdenes")

    # scatter: delivery_time vs customer_rating (relación más interesante)
    sns.scatterplot(data=df_o, x="delivery_time_days", y="customer_rating",
                    hue="order_status", palette=PALETTE_STATUS,
                    alpha=0.5, ax=axes[1])
    axes[1].set_title("Días de entrega vs. Rating (por estado de orden)")
    axes[1].set_xlabel("Días de entrega")
    axes[1].set_ylabel("Rating del cliente")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Análisis de churn
# ══════════════════════════════════════════════════════════════════════════════
def seccion_3(df_c: pd.DataFrame) -> None:
    section_header(3, "Análisis de churn — ¿Quién se va y por qué?")

    churned     = df_c[df_c["churned"] == "Yes"]
    no_churned  = df_c[df_c["churned"] == "No"]
    churn_rate  = len(churned) / len(df_c) * 100

    print(f"\n  Total clientes  : {len(df_c):,}")
    print(f"  Churned (Yes)   : {len(churned):,}  ({churn_rate:.1f}%)")
    print(f"  Activos (No)    : {len(no_churned):,}  ({100 - churn_rate:.1f}%)")

    # Diferencias de medias por grupo
    compare_cols = [
        "avg_order_value", "total_orders", "days_since_last_purchase",
        "engagement_score", "satisfaction_score", "return_rate",
        "cart_abandonment_rate", "discount_usage_rate", "customer_support_tickets",
    ]
    comparison = pd.DataFrame({
        "Activos (media)": no_churned[compare_cols].mean(),
        "Churned (media)": churned[compare_cols].mean(),
    }).round(3)
    comparison["Diferencia %"] = (
        (comparison["Churned (media)"] - comparison["Activos (media)"])
        / comparison["Activos (media)"] * 100
    ).round(1)
    print("\n  Comparación de medias por grupo de churn:")
    print(comparison.to_string())

    # Insights
    biggest_diff = comparison["Diferencia %"].abs().idxmax()
    diff_val     = comparison.loc[biggest_diff, "Diferencia %"]
    insight(
        f"La variable con mayor diferencia entre grupos es '{biggest_diff}' "
        f"({diff_val:+.1f}% en churned vs activos). Esta es la señal más fuerte "
        f"de alerta temprana de fuga — clave para el modelo predictivo."
    )
    insight(
        f"Los clientes churned tienen en promedio "
        f"{churned['days_since_last_purchase'].mean():.0f} días desde su última compra, "
        f"vs {no_churned['days_since_last_purchase'].mean():.0f} días en activos. "
        f"La inactividad es un predictor directo de abandono."
    )
    insight(
        f"El engagement score promedio de churned es "
        f"{churned['engagement_score'].mean():.2f} vs "
        f"{no_churned['engagement_score'].mean():.2f} en activos. "
        f"Un cliente desenganchado es un cliente en riesgo."
    )

    # ── Figura 3A: distribuciones comparativas ────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Sección 3A — Churn: Distribuciones comparativas (Activos vs Churned)",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_cols = [
        ("avg_order_value",          "Valor promedio de orden ($)"),
        ("total_orders",             "Total órdenes"),
        ("days_since_last_purchase", "Días desde última compra"),
        ("engagement_score",         "Engagement score"),
        ("satisfaction_score",       "Satisfaction score"),
        ("return_rate",              "Tasa de devolución"),
        ("cart_abandonment_rate",    "Tasa abandono carrito"),
        ("discount_usage_rate",      "Uso de descuentos"),
        ("customer_support_tickets", "Tickets de soporte"),
    ]
    for idx, (col, label) in enumerate(plot_cols):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        sns.kdeplot(data=df_c, x=col, hue="churned", palette=PALETTE_CHURN,
                    fill=True, alpha=0.4, ax=ax, warn_singular=False)
        ax.set_title(label, fontsize=9.5)
        ax.set_xlabel("")
        ax.legend(title="Churned", fontsize=7, title_fontsize=7)

    plt.show()

    # ── Figura 3B: churn por segmentos ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Sección 3B — Churn por segmentos", fontsize=14, fontweight="bold")

    # Por loyalty
    churn_loyalty = df_c.groupby(["loyalty_status", "churned"]).size().unstack(fill_value=0)
    churn_loyalty_pct = churn_loyalty.div(churn_loyalty.sum(axis=1), axis=0) * 100
    churn_loyalty_pct[["No", "Yes"]].plot(
        kind="bar", color=[PALETTE_CHURN["No"], PALETTE_CHURN["Yes"]],
        ax=axes[0], rot=0, edgecolor="white", width=0.6
    )
    axes[0].set_title("Churn por membresía de lealtad")
    axes[0].set_ylabel("% de clientes")
    axes[0].set_xlabel("")
    axes[0].legend(["Activo", "Churned"], title="Estado")
    for p in axes[0].patches:
        if p.get_height() > 1:
            axes[0].annotate(f"{p.get_height():.1f}%",
                             (p.get_x() + p.get_width()/2, p.get_height()),
                             ha="center", va="bottom", fontsize=8)

    # Por tramo de antigüedad
    df_c["age_group"] = pd.cut(
        df_c["account_age_months"],
        bins=[0, 12, 24, 36, 60, 999],
        labels=["0-12m", "13-24m", "25-36m", "37-60m", "60m+"]
    )
    churn_age = df_c.groupby(["age_group", "churned"], observed=True).size().unstack(fill_value=0)
    churn_age_pct = churn_age.div(churn_age.sum(axis=1), axis=0) * 100
    churn_age_pct["Yes"].plot(kind="bar", color=PALETTE_CHURN["Yes"],
                              ax=axes[1], rot=20, edgecolor="white")
    axes[1].set_title("Tasa de churn por antigüedad de cuenta")
    axes[1].set_ylabel("% churned")
    axes[1].set_xlabel("Tramo de antigüedad")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width()/2, p.get_height()),
                         ha="center", va="bottom", fontsize=8)

    # Boxplot: engagement por churn
    sns.boxplot(data=df_c, x="churned", y="engagement_score",
                palette=PALETTE_CHURN, ax=axes[2], width=0.5)
    axes[2].set_title("Engagement score: Activos vs Churned")
    axes[2].set_xlabel("Churned")
    axes[2].set_ylabel("Engagement score")

    plt.tight_layout()
    plt.show()

    # ── Figura 3C: heatmap de churn por doble segmento ────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Sección 3C — Tasa de churn: Lealtad × Antigüedad",
                 fontsize=13, fontweight="bold")

    pivot_churn = df_c.pivot_table(
        values="churned",
        index="loyalty_status",
        columns="age_group",
        aggfunc=lambda x: (x == "Yes").mean() * 100,
        observed=True,
    ).round(1)

    sns.heatmap(pivot_churn, annot=True, fmt=".1f", cmap="Reds",
                ax=ax, linewidths=0.5, cbar_kws={"label": "% churned"})
    ax.set_title("% de churn por membresía y antigüedad de cuenta")
    ax.set_xlabel("Antigüedad de cuenta")
    ax.set_ylabel("Membresía de lealtad")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Análisis temporal de órdenes
# ══════════════════════════════════════════════════════════════════════════════
def seccion_4(df_o: pd.DataFrame) -> None:
    section_header(4, "Análisis temporal de órdenes")

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    # Resumen mensual
    monthly = (
        df_o.groupby(["year", "month_num", "month_name"])
        .agg(ordenes=("order_id", "count"), revenue=("order_value", "sum"),
             avg_rating=("customer_rating", "mean"))
        .reset_index()
        .sort_values(["year", "month_num"])
    )
    monthly["month_name_short"] = monthly["month_name"].str[:3]

    print("\n  Resumen mensual de órdenes:")
    print(monthly[["year", "month_name", "ordenes", "revenue", "avg_rating"]].to_string(index=False))

    peak_month = monthly.loc[monthly["ordenes"].idxmax()]
    low_month  = monthly.loc[monthly["ordenes"].idxmin()]

    insight(
        f"El mes con más órdenes fue {peak_month['month_name']} {peak_month['year']} "
        f"con {peak_month['ordenes']:.0f} órdenes y revenue de ${peak_month['revenue']:,.0f}. "
        f"El mes más bajo fue {low_month['month_name']} {low_month['year']} "
        f"con {low_month['ordenes']:.0f} órdenes."
    )

    # Día de semana
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = df_o.groupby("day_name").agg(
        ordenes=("order_id", "count"),
        avg_valor=("order_value", "mean")
    ).reindex(day_order)

    best_day = day_stats["ordenes"].idxmax()
    insight(
        f"El día de la semana con más órdenes es {best_day} "
        f"({day_stats.loc[best_day, 'ordenes']:.0f} órdenes en el periodo). "
        f"Los fines de semana representan "
        f"{df_o[df_o['is_weekend']==1]['order_id'].count() / len(df_o) * 100:.1f}% "
        f"del total de órdenes."
    )

    status_dist = df_o["order_status"].value_counts(normalize=True) * 100
    insight(
        f"Del total de órdenes: {status_dist.get('Delivered', 0):.1f}% entregadas, "
        f"{status_dist.get('Cancelled', 0):.1f}% canceladas, "
        f"{status_dist.get('Returned', 0):.1f}% devueltas. "
        f"Una tasa de cancelación + devolución alta sugiere problemas en la experiencia post-compra."
    )

    # ── Figura 4A: tendencia mensual ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Sección 4A — Tendencia mensual de órdenes y revenue",
                 fontsize=14, fontweight="bold")

    x_labels = monthly["month_name_short"].tolist()
    x_pos    = range(len(x_labels))

    axes[0].bar(x_pos, monthly["ordenes"], color=COL_ACCENT, edgecolor="white", width=0.7)
    axes[0].plot(x_pos, monthly["ordenes"], color="#1A3A5C", marker="o", linewidth=2)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels, rotation=30)
    axes[0].set_title("Órdenes por mes")
    axes[0].set_ylabel("Número de órdenes")
    for i, val in enumerate(monthly["ordenes"]):
        axes[0].text(i, val + 1, str(int(val)), ha="center", va="bottom", fontsize=8)

    axes[1].bar(x_pos, monthly["revenue"] / 1_000, color="#6A9E72", edgecolor="white", width=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels, rotation=30)
    axes[1].set_title("Revenue mensual (miles $)")
    axes[1].set_ylabel("Revenue (k$)")

    plt.tight_layout()
    plt.show()

    # ── Figura 4B: órdenes por día y categoría ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sección 4B — Patrones por día de semana y categoría",
                 fontsize=14, fontweight="bold")

    # Día de semana
    palette_days = [COL_ACCENT if d not in ["Saturday", "Sunday"] else "#E8604C" for d in day_order]
    axes[0].bar(day_order, day_stats["ordenes"], color=palette_days, edgecolor="white")
    axes[0].set_title("Órdenes por día de semana\n(rojo = fin de semana)")
    axes[0].set_ylabel("Número de órdenes")
    axes[0].tick_params(axis="x", rotation=30)

    # Categoría × estado (heatmap de revenue promedio)
    pivot_cat = df_o.pivot_table(
        values="order_value", index="product_category",
        columns="order_status", aggfunc="mean"
    ).round(0)
    sns.heatmap(pivot_cat, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=axes[1], linewidths=0.5,
                cbar_kws={"label": "Valor promedio ($)"})
    axes[1].set_title("Valor promedio de orden\n(Categoría × Estado)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

    # ── Figura 4C: rating y entrega por categoría ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sección 4C — Rating y tiempos de entrega por categoría",
                 fontsize=14, fontweight="bold")

    sns.boxplot(data=df_o, x="product_category", y="customer_rating",
                palette="muted", ax=axes[0])
    axes[0].set_title("Rating del cliente por categoría")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=df_o, x="product_category", y="delivery_time_days",
                palette="muted", ax=axes[1])
    axes[1].set_title("Días de entrega por categoría")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log("=== 05 — EDA: Análisis Exploratorio de Datos ===")
    log("Cargando datos desde la DB...")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB en {DB_PATH}. Corre primero los scripts 01–04."
        )

    df_c, df_o = load_data()
    log(f"Clientes: {len(df_c):,} | Órdenes: {len(df_o):,}")
    log("Iniciando análisis. Cierra cada ventana de gráfica para avanzar.\n")

    seccion_1(df_c, df_o)
    seccion_2(df_c, df_o)
    seccion_3(df_c)
    seccion_4(df_o)

    log("✅  EDA completado. Revisa los insights en consola y las gráficas generadas.")


if __name__ == "__main__":
    main()
