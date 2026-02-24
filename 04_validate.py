"""
04_validate.py — Validación de integridad y resumen final
==========================================================
Responsabilidad:
  - Verificar conteos esperados en todas las tablas
  - Detectar FKs rotas en fact tables
  - Mostrar una muestra de las dos vistas principales
  - Dar un resumen de estadísticas descriptivas básicas

Prerequisito: haber corrido 01, 02 y 03.

Cómo correr:
  python 04_validate.py
"""

import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def section(title: str) -> None:
    print()
    print(f"  ╔{'═' * (len(title) + 2)}╗")
    print(f"  ║ {title} ║")
    print(f"  ╚{'═' * (len(title) + 2)}╝")


def main() -> None:
    log("=== 04 — Validación final ===")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB en {DB_PATH}. Corre primero los scripts anteriores."
        )

    conn = sqlite3.connect(DB_PATH)

    # ── 1. Conteos por tabla ──────────────────────────────────────────────────
    section("Conteo de registros por tabla")
    tables = [
        ("fact_customers",        6_000),
        ("fact_orders",           2_600),
        ("dim_loyalty_status",        2),
        ("dim_product_category",      6),
        ("dim_payment_method",        4),
        ("dim_order_status",          3),
        ("dim_date",               None),   # variable, no fijo
    ]
    print(f"\n  {'Tabla':<26} {'Esperado':>10} {'Real':>10} {'OK':>5}")
    print("  " + "─" * 55)
    all_ok = True
    for table, expected in tables:
        real = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if expected is None:
            status = "—"
        elif real == expected:
            status = "✅"
        else:
            status = "❌"
            all_ok = False
        exp_str = f"{expected:,}" if expected else "variable"
        print(f"  {table:<26} {exp_str:>10} {real:>10,} {status:>5}")

    # ── 2. Integridad referencial (FKs rotas) ─────────────────────────────────
    section("Integridad referencial (FKs rotas)")
    fk_checks = [
        ("fact_customers → dim_loyalty", """
            SELECT COUNT(*) FROM fact_customers fc
            LEFT JOIN dim_loyalty_status ls ON fc.loyalty_status_id = ls.loyalty_status_id
            WHERE ls.loyalty_status_id IS NULL
        """),
        ("fact_orders → dim_date", """
            SELECT COUNT(*) FROM fact_orders fo
            LEFT JOIN dim_date dd ON fo.date_id = dd.date_id
            WHERE dd.date_id IS NULL
        """),
        ("fact_orders → dim_category", """
            SELECT COUNT(*) FROM fact_orders fo
            LEFT JOIN dim_product_category pc ON fo.product_category_id = pc.product_category_id
            WHERE pc.product_category_id IS NULL
        """),
        ("fact_orders → dim_payment", """
            SELECT COUNT(*) FROM fact_orders fo
            LEFT JOIN dim_payment_method pm ON fo.payment_method_id = pm.payment_method_id
            WHERE pm.payment_method_id IS NULL
        """),
        ("fact_orders → dim_status", """
            SELECT COUNT(*) FROM fact_orders fo
            LEFT JOIN dim_order_status os ON fo.order_status_id = os.order_status_id
            WHERE os.order_status_id IS NULL
        """),
    ]
    print(f"\n  {'Relación':<35} {'Rotas':>8} {'OK':>5}")
    print("  " + "─" * 50)
    for label, query in fk_checks:
        broken = conn.execute(query).fetchone()[0]
        status = "✅" if broken == 0 else "❌"
        if broken > 0:
            all_ok = False
        print(f"  {label:<35} {broken:>8,} {status:>5}")

    # ── 3. Estadísticas descriptivas básicas ─────────────────────────────────
    section("Estadísticas — Clientes (fact_customers)")
    df_c = pd.read_sql("""
        SELECT avg_order_value, total_orders, days_since_last_purchase,
               engagement_score, satisfaction_score, churned
        FROM fact_customers
    """, conn)
    print()
    print(df_c.describe().round(2).to_string())

    section("Distribución de churn")
    churn = pd.read_sql("""
        SELECT
            CASE churned WHEN 1 THEN 'Churned' ELSE 'Activo' END AS estado,
            COUNT(*) AS clientes,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM fact_customers), 1) AS pct
        FROM fact_customers
        GROUP BY churned
    """, conn)
    print()
    print(churn.to_string(index=False))

    section("Estadísticas — Órdenes (fact_orders)")
    df_o = pd.read_sql("""
        SELECT order_value, delivery_time_days, customer_rating, customer_age
        FROM fact_orders
    """, conn)
    print()
    print(df_o.describe().round(2).to_string())

    section("Órdenes por categoría y estado")
    pivot = pd.read_sql("""
        SELECT
            pc.category_name    AS categoria,
            os.status_name      AS estado,
            COUNT(*)            AS total,
            ROUND(AVG(fo.order_value), 2) AS avg_valor
        FROM fact_orders fo
        JOIN dim_product_category pc ON fo.product_category_id = pc.product_category_id
        JOIN dim_order_status     os ON fo.order_status_id     = os.order_status_id
        GROUP BY pc.category_name, os.status_name
        ORDER BY pc.category_name, os.status_name
    """, conn)
    print()
    print(pivot.to_string(index=False))

    # ── 4. Muestra de vistas ─────────────────────────────────────────────────
    section("Muestra: vw_customers_full (3 filas)")
    print()
    df_vc = pd.read_sql("SELECT * FROM vw_customers_full LIMIT 3", conn)
    print(df_vc.to_string(index=False))

    section("Muestra: vw_orders_full (3 filas)")
    print()
    df_vo = pd.read_sql("SELECT * FROM vw_orders_full LIMIT 3", conn)
    print(df_vo.to_string(index=False))

    # ── Resultado final ───────────────────────────────────────────────────────
    print()
    if all_ok:
        log("✅  Todas las validaciones pasaron. La DB está lista.")
    else:
        log("❌  Algunas validaciones fallaron. Revisa los resultados anteriores.")

    conn.close()


if __name__ == "__main__":
    main()
