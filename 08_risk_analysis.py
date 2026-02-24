"""
08_risk_analysis.py — Análisis de cierre: Segmentos × Riesgo de Churn
======================================================================
Cruza las tres tablas construidas a lo largo del proyecto:
  - fact_customers   → métricas base
  - customer_segments → segmento asignado (Buyer Personas)
  - churn_scores      → score y tier de riesgo (modelo XGBoost)

Responde la pregunta clave:
  ¿Quiénes son los 894 clientes críticos y de qué segmento vienen?

Prerequisito : haber corrido 01–07
Cómo correr  : python 08_risk_analysis.py
"""

import sqlite3
import textwrap
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from config import DB_PATH

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

SEGMENT_COLORS = {
    "💎 Cliente Premium"     : "#4C9BE8",
    "🌟 Cliente Leal"        : "#4CAF50",
    "🏷️  Cazador de Ofertas" : "#FF9800",
    "⚠️  Cliente en Riesgo"  : "#E8604C",
}
TIER_COLORS = {
    "Crítico" : "#E8604C",
    "Alto"    : "#FF9800",
    "Medio"   : "#FFD700",
    "Bajo"    : "#4C9BE8",
}

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def insight(text: str) -> None:
    border = "─" * 62
    wrapped = textwrap.fill(text.strip(), width=60)
    print(f"\n  ┌{border}┐")
    for line in wrapped.splitlines():
        print(f"  │  {line:<60}│")
    print(f"  └{border}┘")

def section_header(title: str) -> None:
    print()
    print("═" * 66)
    print(f"  {title}")
    print("═" * 66)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY PRINCIPAL — cruce de las 3 tablas
# ══════════════════════════════════════════════════════════════════════════════
QUERY_MASTER = """
SELECT
    fc.customer_id,
    cs.segment_name,
    ch.risk_tier,
    ch.churn_score,
    ch.churn_predicted,
    fc.avg_order_value,
    fc.total_orders,
    fc.days_since_last_purchase,
    fc.engagement_score,
    fc.satisfaction_score,
    fc.customer_support_tickets,
    fc.churned,
    ls.status_name AS loyalty_status
FROM fact_customers fc
JOIN dim_loyalty_status  ls ON fc.loyalty_status_id = ls.loyalty_status_id
JOIN customer_segments   cs ON fc.customer_id       = cs.customer_id
JOIN churn_scores        ch ON fc.customer_id       = ch.customer_id
"""

TIER_ORDER    = ["Crítico", "Alto", "Medio", "Bajo"]
SEGMENT_ORDER = list(SEGMENT_COLORS.keys())


def main() -> None:
    log("=== 08 — Análisis de cierre: Segmentos × Riesgo ===\n")

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(QUERY_MASTER, conn)
    conn.close()
    log(f"  Clientes cargados: {len(df):,}")

    # ── 1. Tabla pivot: segmento × tier ──────────────────────────────────────
    section_header("1. Distribución de clientes: Segmento × Tier de riesgo")

    pivot_n = pd.crosstab(df["segment_name"], df["risk_tier"])[TIER_ORDER]
    pivot_pct = pd.crosstab(df["segment_name"], df["risk_tier"],
                             normalize="index")[TIER_ORDER] * 100

    print("\n  Conteo absoluto:")
    print(pivot_n.to_string())
    print("\n  Distribución % dentro de cada segmento:")
    print(pivot_pct.round(1).to_string())

    insight(
        "Esta tabla es el resultado central del proyecto: conecta quién "
        "es el cliente (segmento) con qué tan probable es que se vaya "
        "(tier de riesgo). Es la base para cualquier campaña de retención."
    )

    # ── 2. Foco en los 894 Críticos ───────────────────────────────────────────
    section_header("2. Los clientes Críticos (score ≥ 0.75) por segmento")

    criticos = df[df["risk_tier"] == "Crítico"]
    seg_totals = df["segment_name"].value_counts()

    criticos_seg = (
        criticos.groupby("segment_name")
        .agg(
            clientes       = ("customer_id",             "count"),
            pct_del_total  = ("customer_id",             lambda x: len(x)/len(df)*100),
            avg_score      = ("churn_score",             "mean"),
            avg_dias       = ("days_since_last_purchase","mean"),
            avg_valor      = ("avg_order_value",         "mean"),
            avg_engagement = ("engagement_score",        "mean"),
        )
        .reindex(SEGMENT_ORDER)
        .dropna()
    )
    criticos_seg["pct_del_seg"] = criticos_seg.apply(
        lambda row: row["clientes"] / seg_totals.get(row.name, 1) * 100, axis=1
    )
    criticos_seg = criticos_seg.round(2)

    print(f"\n  Total clientes Críticos: {len(criticos):,}")
    print()
    print(f"  {'Segmento':<28} {'Clientes':>9} {'% total':>8} {'% seg.':>8} "
          f"{'Score':>7} {'Días':>7} {'Valor $':>9} {'Eng.':>6}")
    print("  " + "─" * 82)
    for seg, row in criticos_seg.iterrows():
        print(f"  {seg:<28} {int(row['clientes']):>9,} "
              f"{row['pct_del_total']:>7.1f}% "
              f"{row['pct_del_seg']:>7.1f}% "
              f"{row['avg_score']:>7.3f} "
              f"{row['avg_dias']:>7.0f} "
              f"{row['avg_valor']:>9.2f} "
              f"{row['avg_engagement']:>6.2f}")

    # Hallazgo clave
    seg_mas_criticos = criticos_seg["pct_del_seg"].idxmax()
    pct_max          = criticos_seg.loc[seg_mas_criticos, "pct_del_seg"]
    seg_recuperables = criticos_seg[
        criticos_seg.index.str.contains("Leal|Ofertas|Premium")
    ]["clientes"].sum()

    insight(
        f"El segmento con mayor concentración de críticos es "
        f"'{seg_mas_criticos}' — {pct_max:.1f}% de sus clientes están "
        f"en tier Crítico. Estos ya están prácticamente perdidos."
    )
    insight(
        f"{int(seg_recuperables):,} clientes críticos vienen de segmentos "
        f"Premium, Leal o Cazador de Ofertas — estos SÍ son recuperables "
        f"con intervención inmediata (descuento, llamada, reactivación)."
    )

    # ── 3. Score promedio por segmento × tier ─────────────────────────────────
    section_header("3. Score promedio de churn por segmento")

    score_seg = (
        df.groupby("segment_name")["churn_score"]
        .agg(["mean", "median", "std", "min", "max"])
        .reindex(SEGMENT_ORDER)
        .round(4)
    )
    score_seg.columns = ["Media", "Mediana", "Std", "Min", "Max"]
    print()
    print(score_seg.to_string())

    # ── 4. Visualizaciones ────────────────────────────────────────────────────

    # ── Figura A: heatmap segmento × tier (%) ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Análisis de cierre — Segmentos × Tier de riesgo",
                 fontsize=14, fontweight="bold")

    sns.heatmap(
        pivot_pct.reindex(SEGMENT_ORDER),
        annot=True, fmt=".1f", cmap="RdYlGn_r",
        ax=axes[0], linewidths=0.5,
        cbar_kws={"label": "% del segmento"},
        vmin=0, vmax=100,
    )
    axes[0].set_title("% de clientes por tier dentro de cada segmento")
    axes[0].set_xlabel("Tier de riesgo")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", rotation=0)

    # Stacked bar: composición de cada segmento por tier
    pivot_plot = pivot_pct.reindex(SEGMENT_ORDER)[TIER_ORDER]
    bottom = pd.Series([0.0] * len(pivot_plot), index=pivot_plot.index)
    for tier in TIER_ORDER:
        color = TIER_COLORS[tier]
        axes[1].bar(pivot_plot.index, pivot_plot[tier],
                    bottom=bottom, label=tier, color=color,
                    edgecolor="white", width=0.55)
        # Etiquetas dentro de la barra si el valor es > 5%
        for i, (seg, val) in enumerate(pivot_plot[tier].items()):
            if val > 5:
                axes[1].text(i, bottom[seg] + val/2, f"{val:.0f}%",
                             ha="center", va="center",
                             fontsize=8.5, color="white", fontweight="bold")
        bottom += pivot_plot[tier]

    axes[1].set_title("Composición de riesgo por segmento (100% stacked)")
    axes[1].set_ylabel("% de clientes")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Tier", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # ── Figura B: clientes críticos recuperables vs perdidos ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Clientes Críticos — ¿Recuperables o perdidos?",
                 fontsize=14, fontweight="bold")

    # Barras por segmento
    seg_names   = criticos_seg.index.tolist()
    seg_clientes = criticos_seg["clientes"].tolist()
    colors_bar  = [SEGMENT_COLORS.get(s, "#999") for s in seg_names]

    bars = axes[0].bar(seg_names, seg_clientes,
                       color=colors_bar, edgecolor="white", width=0.6)
    axes[0].set_title(f"Clientes Críticos por segmento (Total: {len(criticos):,})")
    axes[0].set_ylabel("Número de clientes")
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, seg_clientes):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 3,
                     f"{int(val):,}", ha="center", va="bottom", fontsize=9)

    # Pie: recuperables vs ya perdidos
    perdidos      = int(criticos_seg.loc[
        criticos_seg.index.str.contains("Riesgo"), "clientes"
    ].sum()) if any(criticos_seg.index.str.contains("Riesgo")) else 0
    recuperables  = len(criticos) - perdidos

    wedge_data   = [recuperables, perdidos]
    wedge_labels = [f"Recuperables\n{recuperables:,}", f"Ya perdidos\n{perdidos:,}"]
    wedge_colors = ["#4CAF50", "#E8604C"]
    axes[1].pie(wedge_data, labels=wedge_labels, colors=wedge_colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 11})
    axes[1].set_title("Críticos: Recuperables vs Ya perdidos")

    plt.tight_layout()
    plt.show()

    # ── Figura C: scatter engagement vs días sin comprar ─────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.suptitle("Mapa de riesgo: Engagement vs Días sin comprar\n"
                 "(tamaño = score de churn, color = segmento)",
                 fontsize=13, fontweight="bold")

    sample = df.sample(min(1500, len(df)), random_state=42)
    for seg, color in SEGMENT_COLORS.items():
        mask = sample["segment_name"] == seg
        ax.scatter(
            sample.loc[mask, "days_since_last_purchase"],
            sample.loc[mask, "engagement_score"],
            c=color, label=seg,
            s=sample.loc[mask, "churn_score"] * 120 + 10,
            alpha=0.45, edgecolors="none",
        )

    ax.axvline(50,  color="gray", linestyle="--", linewidth=1,
               alpha=0.7, label="50 días sin comprar")
    ax.axhline(3.5, color="gray", linestyle=":",  linewidth=1,
               alpha=0.7, label="Engagement bajo (3.5)")
    ax.set_xlabel("Días desde última compra", fontsize=12)
    ax.set_ylabel("Engagement score", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()

    # ── 5. Resumen ejecutivo final ────────────────────────────────────────────
    section_header("RESUMEN EJECUTIVO DEL PROYECTO")

    total         = len(df)
    n_criticos    = len(criticos)
    n_recuperables = recuperables
    mejor_seg     = df.groupby("segment_name")["churn_score"].mean().idxmin()
    peor_seg      = df.groupby("segment_name")["churn_score"].mean().idxmax()

    print(f"""
  Base de clientes analizada : {total:,}
  Tasa de churn real         : {df['churned'].mean()*100:.1f}%
  Clientes en riesgo Crítico : {n_criticos:,} ({n_criticos/total*100:.1f}%)
  Clientes recuperables      : {n_recuperables:,} ({n_recuperables/total*100:.1f}%)

  Segmento más saludable     : {mejor_seg}
  Segmento más en riesgo     : {peor_seg}

  Mejor modelo predictivo    : Logistic Regression (AUC = 0.9931)
  Variables más importantes  : days_since_last_purchase, engagement_score
    """)

    insight(
        "El proyecto construyó un pipeline completo: desde CSVs crudos "
        "hasta un modelo predictivo con scores de riesgo por cliente. "
        "Siguiente paso recomendado: conectar la DB a Power BI para "
        "un dashboard ejecutivo de retención."
    )

    log("✅  Análisis de cierre completado.")


if __name__ == "__main__":
    main()