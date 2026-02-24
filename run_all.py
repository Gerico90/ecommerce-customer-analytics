"""
run_all.py — Orquestador del pipeline ETL completo
===================================================
Corre los 4 pasos en secuencia. Úsalo cuando ya hayas
validado cada script individualmente.

Cómo correr:
  python run_all.py
"""

import importlib
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_step(module_name: str, step_num: int, description: str) -> None:
    print()
    print("=" * 55)
    log(f"PASO {step_num}: {description}")
    print("=" * 55)
    module = importlib.import_module(module_name)
    module.main()


def main() -> None:
    start = datetime.now()
    log("🚀  Iniciando pipeline ETL completo")

    run_step("01_create_schema", 1, "Crear esquema (tablas, índices, vistas)")
    run_step("02_load_customers", 2, "Cargar clientes")
    run_step("03_load_orders",    3, "Cargar órdenes")
    run_step("04_validate",       4, "Validar integridad")

    elapsed = (datetime.now() - start).total_seconds()
    print()
    print("=" * 55)
    log(f"✅  Pipeline completado en {elapsed:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()
