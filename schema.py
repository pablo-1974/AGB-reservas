# schema.py — creación del esquema PostgreSQL para la aplicación

from db import get_conn

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # Tabla de aulas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        """)

        # Tabla de usuarios
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL CHECK (role IN ('profesor','admin')),
            status TEXT NOT NULL CHECK(status IN ('activo','suspendido')) DEFAULT 'activo',
            password_hash TEXT
        );
        """)

        # Tabla de reservas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id SERIAL PRIMARY KEY,
            room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
            fecha DATE NOT NULL,
            slot_index INTEGER NOT NULL,
            reserved_by_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
            notes TEXT,
            created_at TIMESTAMP NOT NULL,
            UNIQUE(room_id, fecha, slot_index),
            UNIQUE(fecha, slot_index, reserved_by_id)
        );
        """)

        conn.commit()

        # Inserta aulas predeterminadas
        aulas = [
            "Informática A (206)",
            "Informática B (210)",
            "Informática C (209)",
            "Biblioteca",
        ]
        for a in aulas:
            cur.execute("INSERT INTO rooms(name) VALUES (%s) ON CONFLICT DO NOTHING;", (a,))

        conn.commit()
