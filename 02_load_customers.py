"""
02_load_customers.py — Carga del bloque de clientes
=====================================================
Responsabilidad:
  - Leer y unir ecommerce_customer_features + ecommerce_customer_targets
  - Poblar dim_loyalty_status
  - Poblar fact_customers

Prerequisito: haber corrido 01_create_schema.py

Cómo correr:
  python 02_load_customers.py

Qué verificar después:
  SELECT COUNT(*) FROM fact_customers;           -- debe dar 6000
  SELECT COUNT(*) FROM dim_loyalty_status;       -- debe dar 2
  SELECT * FROM vw_customers_full LIMIT 5;
"""

import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH, CSV_FEATURES, CSV_TARGETS


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_dim_loyalty(conn: sqlite3.Connection) -> dict[str, int]:
    """Inserta los dos valores posibles y devuelve el mapa {status_name: id}."""
    conn.executemany(
        "INSERT OR IGNORE INTO dim_loyalty_status (status_name, description) VALUES (?, ?)",
        [
            ("Yes", "Loyalty Member"),
            ("No",  "Regular Customer"),
        ],
    )
    conn.commit()
    rows = conn.execute(
        "SELECT status_name, loyalty_status_id FROM dim_loyalty_status"
    ).fetchall()
    return {name: pk for name, pk in rows}


def load_fact_customers(conn: sqlite3.Connection, loyalty_map: dict[str, int]) -> int:
    """
    Une los dos CSVs, transforma tipos y carga fact_customers.
    Devuelve el número de filas insertadas.
    """
    df_features = pd.read_csv(CSV_FEATURES)
    df_targets  = pd.read_csv(CSV_TARGETS)

    # Unir por Customer_ID (inner join: solo clientes presentes en ambos CSVs)
    df = df_features.merge(df_targets, on="Customer_ID", how="inner")
    log(f"  Filas tras merge: {len(df):,}")

    # Convertir booleanos textuales a enteros
    df["churned"] = (df["churned"] == "Yes").astype(int)

    # Reemplazar loyalty_member (texto) por su FK
    df["loyalty_status_id"] = df["loyalty_member"].map(loyalty_map)

    # Seleccionar y renombrar columnas que van a la DB
    df_insert = df[[
        "Customer_ID", "loyalty_status_id",
        "account_age_months", "total_orders", "days_since_last_purchase",
        "avg_order_value", "price_sensitivity_index",
        "browsing_frequency_per_week", "cart_abandonment_rate",
        "discount_usage_rate", "return_rate",
        "customer_support_tickets", "product_review_score_avg",
        "engagement_score", "satisfaction_score",
        "churned",
    ]].rename(columns={"Customer_ID": "customer_id"})

    df_insert.to_sql("fact_customers", conn, if_exists="append", index=False)
    return len(df_insert)


def main() -> None:
    log("=== 02 — Cargando clientes ===")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB en {DB_PATH}. Corre primero 01_create_schema.py"
        )

    conn = sqlite3.connect(DB_PATH)

    # ── dim_loyalty_status ──────────────────────────────────────────────────
    log("Cargando dim_loyalty_status...")
    loyalty_map = load_dim_loyalty(conn)
    log(f"  Registros: {len(loyalty_map)} → {loyalty_map}")

    # ── fact_customers ───────────────────────────────────────────────────────
    log("Cargando fact_customers...")
    n = load_fact_customers(conn, loyalty_map)
    log(f"  Registros insertados: {n:,}")

    # ── Verificación rápida ──────────────────────────────────────────────────
    print()
    checks = [
        ("Total clientes",    "SELECT COUNT(*) FROM fact_customers"),
        ("Churned = 1",       "SELECT COUNT(*) FROM fact_customers WHERE churned = 1"),
        ("Churned = 0",       "SELECT COUNT(*) FROM fact_customers WHERE churned = 0"),
        ("FK rotas",          """
            SELECT COUNT(*) FROM fact_customers fc
            LEFT JOIN dim_loyalty_status ls ON fc.loyalty_status_id = ls.loyalty_status_id
            WHERE ls.loyalty_status_id IS NULL
        """),
    ]
    print(f"  {'Check':<20} {'Resultado':>10}")
    print("  " + "─" * 32)
    for label, query in checks:
        val = conn.execute(query).fetchone()[0]
        print(f"  {label:<20} {val:>10,}")
    print()

    conn.close()
    log("✅  Bloque clientes cargado correctamente.")


if __name__ == "__main__":
    main()
