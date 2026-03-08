"""Valida conexión a Postgres con las mismas variables que usa ocr_chunking.py."""
import os
import sys

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 no instalado. Ejecuta: pip install psycopg2-binary")
    sys.exit(1)

host = os.getenv("OCR_DB_HOST", "localhost")
port = int(os.getenv("OCR_DB_PORT", "5432"))
dbname = os.getenv("OCR_DB_NAME", "niledb")
user = os.getenv("OCR_DB_USER", "postgres")
password = os.getenv("OCR_DB_PASSWORD", "")

print(f"Conectando a {host}:{port} db={dbname} user={user} ...")
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=5,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user, version();")
        row = cur.fetchone()
    conn.close()
    print("OK - Conexión exitosa.")
    print(f"  Base de datos: {row[0]}")
    print(f"  Usuario:       {row[1]}")
    print(f"  Version:      {row[2].split(',')[0]}")
except Exception as e:
    print(f"ERROR - No se pudo conectar: {e}")
    sys.exit(1)
