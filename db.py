# db.py — conexión global PostgreSQL (Neon + Render)
import os
import psycopg2
import psycopg2.extras

def get_conn():
    return psycopg2.connect(
        os.environ["DATABASE_URL"],
        sslmode="require",
        cursor_factory=psycopg2.extras.DictCursor
    )
