"""
config.py — Configuración centralizada del pipeline ETL
========================================================
Ajusta las rutas de los CSVs según tu entorno local.
El resto del pipeline lee todo desde aquí.
"""

from pathlib import Path

# ── Directorio raíz del proyecto ─────────────────────────────────────────────
# Por defecto apunta a la carpeta donde vive este archivo.
# Cámbialo si quieres guardar la DB en otro lugar.
BASE_DIR = Path(__file__).parent

# ── Rutas de los CSVs fuente ──────────────────────────────────────────────────
# ⚠️  Estos son los únicos valores que debes cambiar en tu máquina local.
CSV_FEATURES = Path("C:/Users/fenix/OneDrive/Escritorio/ecommerce_db/ecommerce_customer_features.csv")
CSV_TARGETS  = Path("C:/Users/fenix/OneDrive/Escritorio/ecommerce_db/ecommerce_customer_targets.csv")
CSV_ORDERS   = Path("C:/Users/fenix/OneDrive/Escritorio/ecommerce_db/daily_ecommerce_orders.csv")

# ── Ruta de la base de datos SQLite ──────────────────────────────────────────
DB_PATH      = BASE_DIR / "ecommerce.db"
