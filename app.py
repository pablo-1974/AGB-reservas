# app.py — versión PostgreSQL completa

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from io import BytesIO
import os
import re
import base64
import hmac
import hashlib
from pathlib import Path

from db import get_conn
from schema import init_db

# ============
# Config
# ============

# Franjas horarias
SLOTS = [
    ("08:40", "09:30"),
    ("09:35", "10:25"),
    ("10:30", "11:20"),
    ("11:50", "12:40"),
    ("12:45", "13:35"),
    ("13:40", "14:30"),
]

DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

PBKDF2_ITERS = 200_000

# ====== Utilidades ======

def lunes_de_semana(d: date) -> date:
    return d - timedelta(days=d.weekday())

def fechas_semana(lunes: date):
    return [lunes + timedelta(days=i) for i in range(5)]

def fin_de_curso(hoy: date) -> date:
    if hoy.month >= 7:
        return date(hoy.year + 1, 6, 30)
    return date(hoy.year, 6, 30)

def hash_password_secure(pwd: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, PBKDF2_ITERS)
    return base64.b64encode(b"pbkdf2$" + salt + dk).decode("ascii")

def verify_password(pwd: str, stored: str) -> bool:
    if stored is None:
        return False
    try:
        raw = base64.b64decode(stored.encode("ascii"))
        if not raw.startswith(b"pbkdf2$"):
            raise ValueError()
        raw = raw[len(b"pbkdf2$"):]
        salt, dk = raw[:16], raw[16:]
        new_dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, PBKDF2_ITERS)
        return hmac.compare_digest(dk, new_dk)
    except Exception:
        return False

# ======================================
#   USUARIOS — PostgreSQL
# ======================================
from db import get_conn
import re

# ----------------------
# ¿Hay usuarios?
# ----------------------
def no_users_exist() -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users;")
        return cur.fetchone()[0] == 0

# ----------------------
# Obtener usuario por email
# ----------------------
def get_user_by_email(email: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, email, role, status, password_hash
            FROM users
            WHERE email=%s
        """, (email.lower().strip(),))
        return cur.fetchone()

# ----------------------
# Crear usuario
# ----------------------
def create_user(name: str, email: str, role: str, status: str = "activo"):
    name = (name or "").strip()
    email = (email or "").strip().lower()
    role = (role or "").strip().lower()
    status = (status or "").strip().lower()

    if not name or not email:
        return False, "Nombre y email son obligatorios."

    if role not in ("profesor", "admin"):
        return False, "Rol no válido."

    if status not in ("activo", "suspendido"):
        return False, "Estado no válido."

    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (name, email, role, status, password_hash)
                VALUES (%s, %s, %s, %s, NULL)
            """, (name, email, role, status))
            conn.commit()
            return True, "Usuario creado correctamente."
        except Exception:
            return False, "Ese email ya está registrado."

# ----------------------
# Lista de todos los usuarios
# ----------------------
def list_users():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, email, role, status, password_hash
            FROM users
            ORDER BY role DESC, name
        """)
        return cur.fetchall()

# ----------------------
# Lista de profesores
# ----------------------
def list_profesores(include_suspended: bool = False):
    with get_conn() as conn:
        cur = conn.cursor()
        if include_suspended:
            cur.execute("""
                SELECT id, name, email
                FROM users
                WHERE role='profesor'
                ORDER BY name
            """)
        else:
            cur.execute("""
                SELECT id, name, email
                FROM users
                WHERE role='profesor' AND status='activo'
                ORDER BY name
            """)
        return cur.fetchall()

# ----------------------
# Establecer contraseña
# ----------------------
def set_user_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET password_hash=%s
            WHERE id=%s
        """, (password_hash, user_id))
        conn.commit()

# ----------------------
# Cambiar estado de usuario
# ----------------------
def set_user_status(user_id: int, status: str):
    if status not in ("activo", "suspendido"):
        return False, "Estado no válido."

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET status=%s
            WHERE id=%s
        """, (status, user_id))
        conn.commit()

    return True, "Estado actualizado."

# ======================================
#   RESERVAS — PostgreSQL
# ======================================

from db import get_conn
from datetime import datetime

# ----------------------
# Listado de reservas por aula y rango de fechas
# ----------------------
def list_reservations(room_id, start_date, end_date):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, room_id, fecha, slot_index, reserved_by_id, notes, created_at
            FROM reservations
            WHERE room_id = %s
              AND fecha BETWEEN %s AND %s
            ORDER BY fecha, slot_index
        """, (room_id, start_date, end_date))
        return cur.fetchall()

# ----------------------
# ¿Profesor ya tiene reserva ese día y franja?
# ----------------------
def profesor_tiene_reserva(fecha, slot_index, profesor_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 1
            FROM reservations
            WHERE fecha = %s
              AND slot_index = %s
              AND reserved_by_id = %s
        """, (fecha, slot_index, profesor_id))
        return cur.fetchone() is not None

# ----------------------
# ¿Existe un conflicto de aula?
# ----------------------
def has_conflict(room_id, fecha, slot_index) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 1
            FROM reservations
            WHERE room_id = %s
              AND fecha = %s
              AND slot_index = %s
        """, (room_id, fecha, slot_index))
        return cur.fetchone() is not None

# ----------------------
# Crear reserva
# ----------------------
def create_reservation(room_id, fecha, slot_index, reserved_by_id, notes=""):
    # Validación rápida en Python
    if profesor_tiene_reserva(fecha, slot_index, reserved_by_id):
        return False, "Ese profesor ya tiene una reserva en esa franja horaria."

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO reservations (
                    room_id, fecha, slot_index, reserved_by_id, notes, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (room_id, fecha, slot_index, reserved_by_id, notes, datetime.now()))
            conn.commit()
        return True, "Reserva creada."
    except Exception:
        return False, "Ya existe una reserva en esa franja para este aula."

# ----------------------
# Borrar reserva
# ----------------------
def delete_reservation(reservation_id):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM reservations WHERE id=%s", (reservation_id,))
        conn.commit()

# ======================================
#   CUADRANTE / DATAFRAME — PostgreSQL
# ======================================

from datetime import date, timedelta
import pandas as pd
from db import get_conn
#from usuarios import list_users
#from reservas import list_reservations
# (Ajusta los imports si usas módulos, o elimina si está todo en app.py)

# ----------------------
# Construir DataFrame semántico semanal
# ----------------------
def build_availability_df(room_id, monday: date):
    """
    Devuelve un DataFrame con tuplas ("LIBRE"|"RESERVADO", "Profesor")
    usado por Streamlit, Excel y PDF.
    """
    # Fechas de lunes — viernes
    dias = [monday + timedelta(days=i) for i in range(5)]

    # Reservas desde PostgreSQL
    reservas = list_reservations(room_id, dias[0], dias[-1])

    # Mapa de usuarios id -> nombre
    users_all = list_users()
    users_map = {row[0]: row[1] for row in users_all}  # (id, name)

    # indexamos reservas por (fecha_iso, slot)
    booked = {}
    for (res_id, room_id, fecha, slot_index, reserved_by_id, notes, created_at) in reservas:
        booked[(fecha, slot_index)] = (reserved_by_id, notes)

    # Preparamos estructura del DF
    idx = [f"{s}-{e}" for s, e in SLOTS]
    cols = [
        f"{DIAS_ES[i]}\n{dias[i].strftime('%d/%m')}"
        for i in range(5)
    ]

    data = []

    for slot_idx in range(len(SLOTS)):
        fila = []
        for d in dias:
            key = (d, slot_idx)
            if key in booked:
                reserved_by_id, notes = booked[key]
                nombre_prof = users_map.get(reserved_by_id, "—")
                fila.append(("RESERVADO", nombre_prof))
            else:
                fila.append(("LIBRE", ""))
        data.append(fila)

    df = pd.DataFrame(data, index=idx, columns=cols)
    return df


# ----------------------
# Convertir DataFrame semántico a plano (para exportar)
# ----------------------
def df_semantico_a_plano(df_sem):
    """
    Convierte un DF con tuplas ("RESERVADO"/"LIBRE", nombre)
    en uno solo de texto ("Libre" o "Nombre").
    """
    df2 = df_sem.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(
            lambda x: x[1] if x[0] == "RESERVADO" else "Libre"
        )
    return df2

# ======================================
#   IMPORTACIÓN DE PROFESORES — PostgreSQL
# ======================================

import pandas as pd
import re
from db import get_conn

def import_profesores_from_excel(file) -> tuple[int, int, list]:
    """
    Importa profesores desde un Excel con columnas 'Nombre' y 'Email'.
    - Crea nuevos usuarios (rol = profesor)
    - Actualiza profesores existentes (nombre + status='activo')
    - Mantiene admins sin cambios
    Devuelve: (creados, actualizados, errores[])
    """

    # ----------------------
    # Leer Excel
    # ----------------------
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as ex:
        return 0, 0, [f"No se pudo leer el Excel: {ex}"]

    # Normalizar nombres de columnas
    cols = {c.strip().lower(): c for c in df.columns}
    if "nombre" not in cols or "email" not in cols:
        return 0, 0, ["El Excel debe contener columnas 'Nombre' y 'Email'."]

    df = df.rename(columns={
        cols["nombre"]: "Nombre",
        cols["email"]: "Email"
    })
    df = df[["Nombre", "Email"]].copy()

    creados = 0
    actualizados = 0
    errores = []

    # Regex de email válido
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    # ----------------------
    # Procesar fila por fila
    # ----------------------
    with get_conn() as conn:
        cur = conn.cursor()

        for idx, row in df.iterrows():
            name = str(row["Nombre"]).strip()
            email = str(row["Email"]).strip().lower()

            # Validación básica
            if not name or not email:
                errores.append(f"Fila {idx+2}: nombre o email vacío.")
                continue

            if not email_re.match(email):
                errores.append(f"Fila {idx+2}: email inválido '{email}'.")
                continue

            # ¿Existe ya el usuario en BD?
            cur.execute("SELECT id, role FROM users WHERE email=%s", (email,))
            existing = cur.fetchone()

            if existing:
                uid, role = existing

                # Solo profesores se pueden actualizar (como en tu app original)
                if role == "profesor":
                    cur.execute("""
                        UPDATE users
                        SET name=%s, status='activo'
                        WHERE id=%s
                    """, (name, uid))
                    actualizados += 1
                else:
                    # Admin: no tocamos nombre ni rol
                    actualizados += 1

            else:
                # Crear profesor nuevo
                try:
                    cur.execute("""
                        INSERT INTO users (name, email, role, status, password_hash)
                        VALUES (%s, %s, 'profesor', 'activo', NULL)
                    """, (name, email))
                    creados += 1
                except Exception:
                    errores.append(f"Fila {idx+2}: email duplicado '{email}'.")

        conn.commit()

    return creados, actualizados, errores


def download_profesores_template_bytes():
    """
    Genera un Excel con la estructura correcta:
    Nombre | Email
    """
    df = pd.DataFrame([
        {"Nombre": "Nombre Apellido", "Email": "nombre.apellido@centro.es"}
    ])
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Profesores")
    out.seek(0)
    return out.read(), "profesores_template.xlsx"

# ======================================
#   EXPORTACIONES — EXCEL y PDF
# ======================================

import pandas as pd
from io import BytesIO

# Intentar cargar ReportLab (si no está instalado, simplemente no se usa)
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


# ----------------------
# Exportar semana a Excel
# ----------------------
def export_week_to_excel_bytes(df_sem, room_name: str, week_monday: date):
    """
    Recibe DataFrame semántico (tuplas) y lo exporta a Excel plano.
    Devuelve: bytes, filename
    """
    # Convertimos ("RESERVADO", "Nombre") → "Nombre" | "Libre"
    df2 = df_semantico_a_plano(df_sem)

    out = BytesIO()
    fname = f"cuadrante_{room_name}_{week_monday}.xlsx"

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df2.to_excel(writer, index=True, sheet_name="Cuadrante")
        ws = writer.book["Cuadrante"]

        # Congelar paneles
        ws.freeze_panes = "B2"

        # Ajustar anchos de columnas automáticamente
        for col in ws.columns:
            try:
                max_len = max(len(str(c.value)) if c.value else 0 for c in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = min(max_len + 2, 30)
            except Exception:
                pass

    out.seek(0)
    return out.read(), fname


# ----------------------
# Exportar semana a PDF
# ----------------------
def export_week_to_pdf_bytes(df_sem, room_name: str, week_monday: date):
    """
    Exporta el cuadrante semanal a PDF si reportlab está disponible.
    Devuelve: bytes, filename
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab no está instalado en este entorno.")

    df2 = df_semantico_a_plano(df_sem)
    buf = BytesIO()
    fname = f"cuadrante_{room_name}_{week_monday}.pdf"

    # PDF horizontal
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter))

    # Encabezados y datos
    headers = ["Hora"] + list(df2.columns)
    data = [headers]

    for idx, row in df2.iterrows():
        data.append([idx] + list(row))

    # Tabla PDF
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.black),
        ("GRID",        (0,0), (-1,-1), 1, colors.black),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
    ]))

    doc.build([table])

    return buf.getvalue(), fname

# ======================================
#   ESTADÍSTICAS DE USO — PostgreSQL
# ======================================

import pandas as pd
from db import get_conn

def obtener_estadisticas():
    """
    Devuelve un diccionario con:
    - raw: DataFrame completo de reservas
    - por_aula: Serie con reservas por aula
    - por_profesor: Serie con reservas por profesor
    - por_dia: Serie con reservas por día de la semana
    - por_franja: Serie con reservas por franja horaria
    """

    # ----------------------
    # Cargar reservas desde PostgreSQL
    # ----------------------
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM reservations", conn)

    if df.empty:
        return None

    # ----------------------
    # Convertir tipos
    # ----------------------
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.weekday  # 0 = lunes

    map_dias = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }
    df["dia_nombre"] = df["dia_semana"].map(map_dias)

    # ----------------------
    # Nombres de aulas desde PostgreSQL
    # ----------------------
    with get_conn() as conn:
        salas = pd.read_sql_query("SELECT id, name FROM rooms", conn)

    aulas_map = dict(zip(salas["id"], salas["name"]))
    df["aula_nombre"] = df["room_id"].map(aulas_map)

    # ----------------------
    # Nombres de profesores desde PostgreSQL
    # ----------------------
    with get_conn() as conn:
        users_df = pd.read_sql_query(
            "SELECT id AS uid, name AS uname FROM users",
            conn
        )

    df = df.merge(users_df, left_on="reserved_by_id", right_on="uid", how="left")

    # ----------------------
    # Estadísticas
    # ----------------------
    reservas_por_aula = (
        df.groupby("aula_nombre")["id"]
        .count()
        .sort_values(ascending=False)
    )

    reservas_por_profesor = (
        df.groupby("uname")["id"]
        .count()
        .sort_values(ascending=False)
    )

    # Solo días lectivos
    reservas_por_dia = (
        df[df["dia_semana"] <= 4]
        .groupby("dia_nombre")["id"]
        .count()
        .reindex(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"])
    )

    # Etiquetas de franja
    map_slots = {i: f"{SLOTS[i][0]}–{SLOTS[i][1]}" for i in range(len(SLOTS))}
    df["slot_label"] = df["slot_index"].map(map_slots)

    reservas_por_franja = (
        df.groupby("slot_label")["id"]
        .count()
        .reindex(list(map_slots.values()))
    )

    # ----------------------
    # Devolver todo empaquetado
    # ----------------------
    return {
        "raw": df,
        "por_aula": reservas_por_aula,
        "por_profesor": reservas_por_profesor,
        "por_dia": reservas_por_dia,
        "por_franja": reservas_por_franja,
    }

# ======================================
# BLOQUE 10.1 — AUTENTICACIÓN STREAMLIT
# ======================================

def login_screen():
    st.title("🔐 Acceso")
    email = st.text_input("Email institucional", key="login_email")

    if st.button("Continuar", key="login_continue"):
        u = get_user_by_email(email)
        if not u:
            st.error("Email no registrado.")
            return

        uid, name, email, role, status, pwd_hash = u

        if role == "profesor" and status != "activo":
            st.error("Tu cuenta está suspendida. Contacta con un administrador.")
            return

        if pwd_hash is None:
            st.session_state["pending_user"] = {
                "id": uid,
                "name": name,
                "email": email,
                "role": role,
                "status": status
            }
            st.session_state["needs_password_setup"] = True
            st.rerun()
        else:
            st.session_state["login_user"] = u
            st.session_state["ask_password"] = True
            st.rerun()


def first_password_screen():
    u = st.session_state["pending_user"]
    st.title("🔑 Crear contraseña nueva")
    st.write(f"Usuario: **{u['name']}** ({u['email']})")

    pwd1 = st.text_input("Nueva contraseña", type="password", key="fp_p1")
    pwd2 = st.text_input("Repetir contraseña", type="password", key="fp_p2")

    if st.button("Guardar contraseña", key="fp_save"):
        if pwd1 != pwd2:
            st.error("Las contraseñas no coinciden.")
            return
        if len(pwd1) < 4:
            st.error("Debe tener al menos 4 caracteres.")
            return

        set_user_password(u["id"], hash_password_secure(pwd1))
        st.success("Contraseña creada. Inicia sesión.")
        st.session_state.clear()
        st.rerun()


def password_login_screen():
    u = st.session_state["login_user"]
    uid, name, email, role, status, pwd_hash = u

    st.title("🔒 Introduce tu contraseña")
    pwd = st.text_input("Contraseña", type="password", key="pl_pass")

    if st.button("Entrar", key="pl_enter"):
        if role == "profesor" and status != "activo":
            st.error("Tu cuenta está suspendida. Contacta con un administrador.")
            return

        if verify_password(pwd, pwd_hash):
            st.session_state["user"] = {
                "id": uid,
                "name": name,
                "email": email,
                "role": role,
                "status": status
            }
            st.session_state.pop("login_user")
            st.session_state.pop("ask_password")
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")


def bootstrap_admin_screen():
    st.title("🛠 Configuración inicial")
    st.write("No hay usuarios. Crea el **primer administrador**.")

    name = st.text_input("Nombre completo", key="bs_name")
    email = st.text_input("Email", key="bs_email")
    p1 = st.text_input("Contraseña", type="password", key="bs_p1")
    p2 = st.text_input("Repetir contraseña", type="password", key="bs_p2")

    if st.button("Crear administrador", key="bs_create"):
        if not name or not email:
            st.error("Nombre y email obligatorios.")
            return
        if p1 != p2:
            st.error("Las contraseñas no coinciden.")
            return
        if len(p1) < 4:
            st.error("Contraseña demasiado corta.")
            return

        ok, msg = create_user(name, email, "admin", "activo")
        if ok:
            uid = get_user_by_email(email)[0]
            set_user_password(uid, hash_password_secure(p1))
            st.success("Administrador creado. Inicia sesión.")
            st.session_state.clear()
            st.rerun()
        else:
            st.error(msg)

# ======================================
# BLOQUE 10.2 — NAVEGACIÓN SEMANAL + AULA + CUADRANTE
# ======================================

def render_week_navigation():
    if "week_monday" not in st.session_state:
        st.session_state["week_monday"] = lunes_de_semana(date.today())

    colL, colC, colR = st.columns([1, 2, 1])

    with colL:
        if st.button("⬅️ Semana anterior", key="prev_week"):
            st.session_state["week_monday"] -= timedelta(days=7)

    with colR:
        if st.button("➡️ Semana siguiente", key="next_week"):
            st.session_state["week_monday"] += timedelta(days=7)

    with colC:
        sel_date = st.date_input(
            "Ir a semana:",
            value=st.session_state["week_monday"],
            key="go_to_week"
        )
        st.session_state["week_monday"] = lunes_de_semana(sel_date)

    return st.session_state["week_monday"]


def render_room_selector():
    rooms = get_rooms()
    room_map = {rid: name for rid, name in rooms}

    room_id = st.selectbox(
        "Aula",
        list(room_map.keys()),
        format_func=lambda r: room_map[r],
        key="room_select"
    )
    return room_id, room_map[room_id]


def render_weekly_grid(room_id, room_name, week_monday):
    week_start = week_monday
    week_end = week_monday + timedelta(days=4)

    st.markdown(
        f"### 🗓 Semana del **{week_start.strftime('%d/%m')}** "
        f"al **{week_end.strftime('%d/%m')}**"
    )

    df_sem = build_availability_df(room_id, week_monday)
    df_disp = df_sem.copy()

    for col in df_disp.columns:
        df_disp[col] = df_disp[col].apply(
            lambda x: f"🟥 {x[1]}" if x[0] == "RESERVADO" else "🟩 Libre"
        )

    st.subheader(f"Aula: {room_name}")
    st.dataframe(df_disp, use_container_width=True)

    return df_sem

# ======================================
# BLOQUE 10.3 — CREAR Y CANCELAR RESERVAS
# ======================================

def render_create_reservation(usuario, room_id, week_monday):
    st.markdown("### ➕ Nueva reserva")

    if usuario["role"] == "profesor":
        reserved_by_id = usuario["id"]
        st.info(f"Profesor: **{usuario['name']}**")
    else:
        profesores = list_profesores(include_suspended=False)
        if not profesores:
            st.error("No hay profesores activos registrados.")
            return

        pm = {pid: f"{n} ({e})" for pid, n, e in profesores}
        reserved_by_id = st.selectbox(
            "Profesor",
            list(pm.keys()),
            format_func=lambda x: pm[x],
            key="reserve_prof"
        )

    day_idx = st.selectbox(
        "Día",
        range(5),
        format_func=lambda i: DIAS_ES[i],
        key="reserve_day"
    )

# ======================================
# BLOQUE 10.4 — RESERVAS RECURRENTES (ADMIN)
# ======================================

def render_recurring_reservations(usuario):
    if usuario["role"] != "admin":
        return

    st.divider()
    st.header("📆 Reservas recurrentes")

    with st.form("form_rec"):
        profesores = list_profesores(include_suspended=False)
        if not profesores:
            st.warning("No hay profesores activos. Importa o crea alguno primero.")
            st.form_submit_button("Crear reservas recurrentes", disabled=True)
            return

        mp = {pid: f"{n} ({e})" for pid, n, e in profesores}
        reserved_rec_id = st.selectbox(
            "Profesor",
            list(mp.keys()),
            format_func=lambda x: mp[x],
            key="rec_pid"
        )

        rm = {rid: name for rid, name in get_rooms()}
        if not rm:
            st.error("No hay aulas registradas.")
            st.form_submit_button("Crear reservas recurrentes", disabled=True)
            return

        room_rec = st.selectbox(
            "Aula",
            list(rm.keys()),
            format_func=lambda r: rm[r],
            key="rec_room"
        )

        day_idx_rec = st.selectbox(
            "Día semanal",
            range(5),
            format_func=lambda i: DIAS_ES[i],
            key="rec_day"
        )

        slot_idx_rec = st.selectbox(
            "Franja horaria",
            range(len(SLOTS)),
            format_func=lambda i: f"{SLOTS[i][0]}–{SLOTS[i][1]}",
            key="rec_slot"
        )

        notes_rec = st.text_input("Notas", key="rec_notes")

        sub_rec = st.form_submit_button("Crear reservas recurrentes", key="rec_submit")

    if sub_rec:
        hoy = date.today()
        fin = fin_de_curso(hoy)
        delta = (day_idx_rec - hoy.weekday()) % 7
        fecha = hoy + timedelta(days=delta)

        creadas = 0
        conflictos = 0

        while fecha <= fin:
            if (not profesor_tiene_reserva(fecha, slot_idx_rec, reserved_rec_id)) and \
               (not has_conflict(room_rec, fecha, slot_idx_rec)):
                ok, _ = create_reservation(room_rec, fecha, slot_idx_rec, reserved_rec_id, notes_rec)
                if ok:
                    creadas += 1
                else:
                    conflictos += 1
            else:
                conflictos += 1
            fecha += timedelta(days=7)

        st.success(f"✔ {creadas} reservas creadas.")
        if conflictos:
            st.warning(f"⚠ {conflictos} conflictos omitidos.")

# ======================================
# BLOQUE 10.5 — GESTIÓN USUARIOS + ESTADÍSTICAS + EXPORT/BACKUP + LAYOUT FINAL
# ======================================

def get_rooms():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM rooms ORDER BY name;")
        return cur.fetchall()


def render_sidebar(usuario):
    with st.sidebar:
        try:
            st.image("logo.png", width=150)
        except Exception:
            pass

        st.markdown("---")
        st.write(f"👤 {usuario['name']} ({usuario['role']})")

        if st.button("Cerrar sesión", key="sidebar_logout"):
            st.session_state.clear()
            st.rerun()

        st.markdown("---")
        st.subheader("🔑 Cambiar contraseña")
        old = st.text_input("Contraseña actual", type="password", key="oldpass")
        new1 = st.text_input("Nueva contraseña", type="password", key="newpass1")
        new2 = st.text_input("Repetir contraseña", type="password", key="newpass2")

        if st.button("Actualizar contraseña", key="sidebar_update_pwd"):
            u = get_user_by_email(usuario["email"])
            if not u:
                st.error("Usuario no encontrado.")
            else:
                if not verify_password(old, u[5]):
                    st.error("❌ La contraseña actual no coincide.")
                elif new1 != new2:
                    st.error("❌ Las nuevas contraseñas no coinciden.")
                elif len(new1) < 4:
                    st.error("❌ Contraseña demasiado corta.")
                else:
                    set_user_password(usuario["id"], hash_password_secure(new1))
                    st.success("✔ Contraseña actualizada.")

        st.markdown("---")


def get_db_backup_zip_bytes_and_name():
    import io, zipfile
    with get_conn() as conn:
        rooms_df = pd.read_sql_query("SELECT * FROM rooms ORDER BY id;", conn)
        users_df = pd.read_sql_query("SELECT * FROM users ORDER BY id;", conn)
        reservations_df = pd.read_sql_query("SELECT * FROM reservations ORDER BY id;", conn)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("rooms.csv", rooms_df.to_csv(index=False))
        zf.writestr("users.csv", users_df.to_csv(index=False))
        zf.writestr("reservations.csv", reservations_df.to_csv(index=False))
    buf.seek(0)
    name = f"backup_reservas_{date.today().isoformat()}.zip"
    return buf.read(), name


def render_admin_exports_and_backup(usuario, df_sem, room_name, week_monday):
    if usuario["role"] != "admin":
        return

    st.markdown("### 📤 Exportar cuadrante / Backup")
    c1, c2, c3 = st.columns(3)

    with c1:
        try:
            xb, xf = export_week_to_excel_bytes(df_sem, room_name, week_monday)
            st.download_button("💾 Excel", xb, xf, key="export_excel")
        except Exception as ex:
            st.error(f"Error exportando a Excel: {ex}")

    with c2:
        try:
            pb, pf = export_week_to_pdf_bytes(df_sem, room_name, week_monday)
            st.download_button("🖨 PDF", pb, pf, key="export_pdf")
        except Exception as ex:
            st.warning("Para PDF instala reportlab: pip install reportlab")
            st.caption(f"Detalle: {ex}")

    with c3:
        try:
            zip_bytes, zip_name = get_db_backup_zip_bytes_and_name()
            st.download_button("🔐 Backup BD", data=zip_bytes, file_name=zip_name,
                               mime="application/zip", key="backup_db_zip")
        except Exception as ex:
            st.error(f"No se pudo generar el backup: {ex}")


def render_admin_user_management(usuario):
    if usuario["role"] != "admin":
        return

    st.divider()
    st.header("👥 Gestión de usuarios")

    st.subheader("📥 Importar profesores desde Excel")
    cta1, cta2 = st.columns([1, 1])

    with cta1:
        template_bytes, template_name = download_profesores_template_bytes()
        st.download_button("📄 Descargar plantilla", template_bytes, template_name, key="dl_template")

    with cta2:
        up = st.file_uploader("Sube el archivo 'profesores.xlsx' (Nombre, Email)", type=["xlsx"], key="upload_prof")
        if up is not None:
            creados, actualizados, errores = import_profesores_from_excel(up)
            st.success(f"Profesores creados: {creados} · actualizados: {actualizados}")
            if errores:
                with st.expander("Ver detalles de errores"):
                    for e in errores:
                        st.write("• " + e)

    st.markdown("---")
    st.subheader("➕ Crear usuario manualmente")
    with st.form("form_new_user", clear_on_submit=True):
        colA, colB, colC, colD = st.columns([2, 2, 1, 1])
        with colA:
            n = st.text_input("Nombre completo", key="new_user_name")
        with colB:
            e = st.text_input("Email", key="new_user_email")
        with colC:
            r = st.selectbox("Rol", ["profesor", "admin"], key="new_user_role")
        with colD:
            s = st.selectbox("Estado", ["activo", "suspendido"], index=0, key="new_user_status")
        if st.form_submit_button("Crear usuario", key="new_user_btn"):
            ok, msg = create_user(n, e, r, s)
            (st.success if ok else st.error)(msg)

    users_raw = list_users()
    df_users = pd.DataFrame(users_raw, columns=["ID", "Nombre", "Email", "Rol", "Estado", "PasswordHash"])
    df_users["TieneContraseña"] = df_users["PasswordHash"].notna()
    df_users = df_users.drop(columns=["PasswordHash"])
    st.dataframe(df_users, hide_index=True, use_container_width=True)

    st.subheader("🚦 Cambiar estado (activo/suspendido)")
    all_users_map = {uid: f"{name} <{email}> [{role}] ({estado})" for uid, name, email, role, estado, pwh in users_raw}
    if all_users_map:
        uid_sel = st.selectbox("Usuario", list(all_users_map.keys()),
                               format_func=lambda k: all_users_map[k], key="status_user_sel")
        new_status = st.selectbox("Nuevo estado", ["activo", "suspendido"], key="status_new")
        if st.button("Actualizar estado", key="btn_update_status"):
            ok, msg = set_user_status(uid_sel, new_status)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    st.subheader("🧹 Resetear contraseña de profesor")
    profs_all = [u for u in users_raw if u[3] == "profesor"]
    if profs_all:
        prof_map = {uid: f"{name} <{email}> ({estado})" for uid, name, email, role, estado, pwh in profs_all}
        psel = st.selectbox("Profesor", list(prof_map.keys()),
                            format_func=lambda k: prof_map[k], key="reset_prof_sel")
        np1 = st.text_input("Nueva contraseña", type="password", key="reset_pw1")
        np2 = st.text_input("Repetir contraseña", type="password", key="reset_pw2")
        if st.button("Resetear contraseña", key="btn_reset_pw"):
            if np1 != np2:
                st.error("Las contraseñas no coinciden.")
            elif len(np1) < 4:
                st.error("Contraseña demasiado corta.")
            else:
                set_user_password(psel, hash_password_secure(np1))
                st.success("Contraseña reseteada correctamente.")
    else:
        st.info("Aún no hay profesores registrados.")


def render_admin_stats(usuario):
    if usuario["role"] != "admin":
        return

    st.divider()
    st.header("📈 Estadísticas de uso")

    est = obtener_estadisticas()
    if est is None:
        st.info("Todavía no hay reservas registradas para generar estadísticas.")
        return

    st.subheader("📊 Reservas por aula")
    st.bar_chart(est["por_aula"])
    st.dataframe(est["por_aula"].reset_index().rename(columns={"id": "Reservas"}))

    st.subheader("👩‍🏫 Reservas por profesor")
    st.bar_chart(est["por_profesor"])
    st.dataframe(est["por_profesor"].reset_index().rename(columns={"id": "Reservas"}))

    st.subheader("🗓 Reservas por día de la semana")
    st.bar_chart(est["por_dia"])
    st.dataframe(est["por_dia"].reset_index().rename(columns={"id": "Reservas"}))

    st.subheader("⏰ Reservas por franja horaria")
    st.bar_chart(est["por_franja"])
    st.dataframe(est["por_franja"].reset_index().rename(columns={"id": "Reservas"}))


def main():
    st.set_page_config(page_title="Reserva de Aulas", layout="wide")
    init_db()

    if no_users_exist():
        bootstrap_admin_screen()
        return

    if "needs_password_setup" in st.session_state:
        first_password_screen()
        return

    if "ask_password" in st.session_state:
        password_login_screen()
        return

    if "user" not in st.session_state:
        login_screen()
        return

    usuario = st.session_state["user"]

    render_sidebar(usuario)

    st.title("📚 Reserva de Aulas — IES Antonio García Bellido")

    week_monday = render_week_navigation()
    room_id, room_name = render_room_selector()
    df_sem = render_weekly_grid(room_id, room_name, week_monday)

    render_admin_exports_and_backup(usuario, df_sem, room_name, week_monday)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_create_reservation(usuario, room_id, week_monday)
    with col2:
        render_cancel_reservation(usuario, room_id, week_monday)

    render_recurring_reservations(usuario)
    render_admin_user_management(usuario)
    render_admin_stats(usuario)


if __name__ == "__main__":
    main()



