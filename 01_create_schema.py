"""
01_create_schema.py — Creación del esquema en SQLite
=====================================================
Responsabilidad: ejecutar el DDL completo.
  - Crea todas las tablas (dim_* y fact_*)
  - Crea índices
  - Crea vistas (vw_*)

Cómo correr:
  python 01_create_schema.py

Qué verificar después:
  Abre ecommerce.db con DB Browser for SQLite (o cualquier cliente SQLite)
  y confirma que aparecen las 7 tablas, los índices y las 2 vistas.
"""

import sqlite3
from datetime import datetime
from config import DB_PATH


DDL = """
PRAGMA foreign_keys = ON;

-- ── BLOQUE CLIENTES ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_loyalty_status (
    loyalty_status_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    status_name         TEXT    NOT NULL UNIQUE,
    description         TEXT
);

CREATE TABLE IF NOT EXISTS fact_customers (
    customer_id                 TEXT    PRIMARY KEY,
    loyalty_status_id           INTEGER NOT NULL REFERENCES dim_loyalty_status(loyalty_status_id),
    account_age_months          INTEGER NOT NULL,
    total_orders                INTEGER NOT NULL,
    days_since_last_purchase    INTEGER NOT NULL,
    avg_order_value             REAL    NOT NULL,
    price_sensitivity_index     REAL    NOT NULL,
    browsing_frequency_per_week REAL    NOT NULL,
    cart_abandonment_rate       REAL    NOT NULL,
    discount_usage_rate         REAL    NOT NULL,
    return_rate                 REAL    NOT NULL,
    customer_support_tickets    INTEGER NOT NULL,
    product_review_score_avg    REAL    NOT NULL,
    engagement_score            REAL    NOT NULL,
    satisfaction_score          REAL    NOT NULL,
    churned                     INTEGER NOT NULL CHECK (churned IN (0, 1))
);

-- ── BLOQUE ÓRDENES ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_product_category (
    product_category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name       TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_payment_method (
    payment_method_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    method_name         TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_order_status (
    order_status_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    status_name         TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id             INTEGER PRIMARY KEY,
    full_date           TEXT    NOT NULL UNIQUE,
    year                INTEGER NOT NULL,
    quarter             INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month               INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name          TEXT    NOT NULL,
    week_of_year        INTEGER NOT NULL,
    day_of_month        INTEGER NOT NULL,
    day_of_week         INTEGER NOT NULL,
    day_name            TEXT    NOT NULL,
    is_weekend          INTEGER NOT NULL CHECK (is_weekend IN (0, 1))
);

CREATE TABLE IF NOT EXISTS fact_orders (
    order_id                INTEGER PRIMARY KEY,
    date_id                 INTEGER NOT NULL REFERENCES dim_date(date_id),
    product_category_id     INTEGER NOT NULL REFERENCES dim_product_category(product_category_id),
    payment_method_id       INTEGER NOT NULL REFERENCES dim_payment_method(payment_method_id),
    order_status_id         INTEGER NOT NULL REFERENCES dim_order_status(order_status_id),
    customer_age            INTEGER NOT NULL,
    order_value             REAL    NOT NULL,
    delivery_time_days      INTEGER NOT NULL,
    customer_rating         REAL    NOT NULL,
    discount_applied        INTEGER NOT NULL CHECK (discount_applied IN (0, 1))
);

-- ── ÍNDICES ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_fc_loyalty  ON fact_customers(loyalty_status_id);
CREATE INDEX IF NOT EXISTS idx_fc_churned  ON fact_customers(churned);
CREATE INDEX IF NOT EXISTS idx_fo_date     ON fact_orders(date_id);
CREATE INDEX IF NOT EXISTS idx_fo_category ON fact_orders(product_category_id);
CREATE INDEX IF NOT EXISTS idx_fo_payment  ON fact_orders(payment_method_id);
CREATE INDEX IF NOT EXISTS idx_fo_status   ON fact_orders(order_status_id);
CREATE INDEX IF NOT EXISTS idx_fo_date_cat ON fact_orders(date_id, product_category_id);

-- ── VISTAS ────────────────────────────────────────────────────────────────────

CREATE VIEW IF NOT EXISTS vw_customers_full AS
SELECT
    fc.customer_id,
    ls.status_name                  AS loyalty_status,
    fc.account_age_months,
    fc.total_orders,
    fc.days_since_last_purchase,
    fc.avg_order_value,
    fc.price_sensitivity_index,
    fc.browsing_frequency_per_week,
    fc.cart_abandonment_rate,
    fc.discount_usage_rate,
    fc.return_rate,
    fc.customer_support_tickets,
    fc.product_review_score_avg,
    fc.engagement_score,
    fc.satisfaction_score,
    CASE fc.churned WHEN 1 THEN 'Yes' ELSE 'No' END AS churned
FROM fact_customers fc
JOIN dim_loyalty_status ls ON fc.loyalty_status_id = ls.loyalty_status_id;

CREATE VIEW IF NOT EXISTS vw_orders_full AS
SELECT
    fo.order_id,
    dd.full_date        AS order_date,
    dd.year,
    dd.month_name,
    dd.day_name,
    dd.is_weekend,
    pc.category_name    AS product_category,
    pm.method_name      AS payment_method,
    os.status_name      AS order_status,
    fo.customer_age,
    fo.order_value,
    fo.delivery_time_days,
    fo.customer_rating,
    CASE fo.discount_applied WHEN 1 THEN 'Yes' ELSE 'No' END AS discount_applied
FROM fact_orders fo
JOIN dim_date             dd ON fo.date_id             = dd.date_id
JOIN dim_product_category pc ON fo.product_category_id = pc.product_category_id
JOIN dim_payment_method   pm ON fo.payment_method_id   = pm.payment_method_id
JOIN dim_order_status     os ON fo.order_status_id     = os.order_status_id;
"""


def main() -> None:
    log("=== 01 — Creando esquema ===")

    # Si la DB ya existe la borramos para empezar limpio
    if DB_PATH.exists():
        DB_PATH.unlink()
        log(f"DB previa eliminada: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(DDL)
    conn.commit()

    # Verificación: listar objetos creados
    cursor = conn.cursor()
    objects = cursor.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table','index','view') ORDER BY type, name"
    ).fetchall()

    print()
    print(f"  {'Tipo':<8} {'Nombre'}")
    print("  " + "─" * 38)
    for obj_type, obj_name in objects:
        print(f"  {obj_type:<8} {obj_name}")
    print()

    conn.close()
    log(f"✅  Esquema creado correctamente en: {DB_PATH}")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    main()
