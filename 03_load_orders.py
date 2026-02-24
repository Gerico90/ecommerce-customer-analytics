"""
03_load_orders.py — Carga del bloque de órdenes
================================================
Responsabilidad:
  - Leer daily_ecommerce_orders.csv
  - Poblar dim_product_category, dim_payment_method, dim_order_status
  - Construir y poblar dim_date
  - Poblar fact_orders

Prerequisito: haber corrido 01_create_schema.py

Cómo correr:
  python 03_load_orders.py

Qué verificar después:
  SELECT COUNT(*) FROM fact_orders;              -- debe dar 2600
  SELECT COUNT(*) FROM dim_date;                 -- debe dar ~364
  SELECT * FROM vw_orders_full LIMIT 5;
"""

import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH, CSV_ORDERS


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def build_lookup(conn: sqlite3.Connection, table: str, pk_col: str,
                 name_col: str, values: list[str]) -> dict[str, int]:
    """
    Inserta valores únicos en una tabla dim y devuelve {valor: pk_id}.
    Usa INSERT OR IGNORE para ser idempotente.
    """
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({name_col}) VALUES (?)",
        [(v,) for v in values],
    )
    conn.commit()
    rows = conn.execute(f"SELECT {name_col}, {pk_col} FROM {table}").fetchall()
    return {name: pk for name, pk in rows}


def build_date_dim(dates: pd.Series) -> pd.DataFrame:
    """
    Construye el DataFrame de dim_date a partir de una serie de fechas.
    La PK (date_id) usa el formato YYYYMMDD como entero.
    """
    month_names = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    records = []
    for d in sorted(pd.to_datetime(dates.unique())):
        records.append({
            "date_id"      : int(d.strftime("%Y%m%d")),
            "full_date"    : d.strftime("%Y-%m-%d"),
            "year"         : d.year,
            "quarter"      : (d.month - 1) // 3 + 1,
            "month"        : d.month,
            "month_name"   : month_names[d.month],
            "week_of_year" : int(d.strftime("%W")),
            "day_of_month" : d.day,
            "day_of_week"  : d.weekday() + 1,   # 1=Monday … 7=Sunday
            "day_name"     : day_names[d.weekday()],
            "is_weekend"   : 1 if d.weekday() >= 5 else 0,
        })
    return pd.DataFrame(records)


def main() -> None:
    log("=== 03 — Cargando órdenes ===")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró la DB en {DB_PATH}. Corre primero 01_create_schema.py"
        )

    conn = sqlite3.connect(DB_PATH)

    # ── Leer CSV ─────────────────────────────────────────────────────────────
    log("Leyendo CSV de órdenes...")
    df = pd.read_csv(CSV_ORDERS)
    df["order_date"] = pd.to_datetime(df["order_date"])
    log(f"  Filas leídas: {len(df):,}")

    # ── Dimensiones categóricas ───────────────────────────────────────────────
    log("Cargando dim_product_category...")
    cat_map = build_lookup(
        conn, "dim_product_category", "product_category_id", "category_name",
        df["product_category"].unique().tolist(),
    )
    log(f"  Categorías: {list(cat_map.keys())}")

    log("Cargando dim_payment_method...")
    pay_map = build_lookup(
        conn, "dim_payment_method", "payment_method_id", "method_name",
        df["payment_method"].unique().tolist(),
    )
    log(f"  Métodos: {list(pay_map.keys())}")

    log("Cargando dim_order_status...")
    status_map = build_lookup(
        conn, "dim_order_status", "order_status_id", "status_name",
        df["order_status"].unique().tolist(),
    )
    log(f"  Estados: {list(status_map.keys())}")

    # ── dim_date ──────────────────────────────────────────────────────────────
    log("Construyendo y cargando dim_date...")
    df_date = build_date_dim(df["order_date"])
    df_date.to_sql("dim_date", conn, if_exists="append", index=False)
    log(f"  Fechas únicas cargadas: {len(df_date):,}")

    date_map = {
        row[0]: row[1]
        for row in conn.execute("SELECT full_date, date_id FROM dim_date").fetchall()
    }

    # ── fact_orders ───────────────────────────────────────────────────────────
    log("Cargando fact_orders...")
    df["date_id"]             = df["order_date"].dt.strftime("%Y-%m-%d").map(date_map)
    df["product_category_id"] = df["product_category"].map(cat_map)
    df["payment_method_id"]   = df["payment_method"].map(pay_map)
    df["order_status_id"]     = df["order_status"].map(status_map)
    df["discount_applied"]    = (df["discount_applied"] == "Yes").astype(int)

    df_insert = df[[
        "order_id", "date_id", "product_category_id",
        "payment_method_id", "order_status_id",
        "customer_age", "order_value", "delivery_time_days",
        "customer_rating", "discount_applied",
    ]]
    df_insert.to_sql("fact_orders", conn, if_exists="append", index=False)
    log(f"  Registros insertados: {len(df_insert):,}")

    # ── Verificación rápida ───────────────────────────────────────────────────
    print()
    checks = [
        ("Total órdenes",       "SELECT COUNT(*) FROM fact_orders"),
        ("dim_date",            "SELECT COUNT(*) FROM dim_date"),
        ("dim_product_cat.",    "SELECT COUNT(*) FROM dim_product_category"),
        ("dim_payment_method",  "SELECT COUNT(*) FROM dim_payment_method"),
        ("dim_order_status",    "SELECT COUNT(*) FROM dim_order_status"),
        ("FK fecha rota",       """
            SELECT COUNT(*) FROM fact_orders fo
            LEFT JOIN dim_date dd ON fo.date_id = dd.date_id
            WHERE dd.date_id IS NULL
        """),
    ]
    print(f"  {'Check':<22} {'Resultado':>10}")
    print("  " + "─" * 34)
    for label, query in checks:
        val = conn.execute(query).fetchone()[0]
        print(f"  {label:<22} {val:>10,}")
    print()

    conn.close()
    log("✅  Bloque órdenes cargado correctamente.")


if __name__ == "__main__":
    main()
