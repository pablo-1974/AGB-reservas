# PRUEBAS
# app.py — versión PostgreSQL completa con mejoras solicitadas

# ==== Imports base ====
import os
import re
import base64
import hmac
import hashlib
from io import BytesIO
from datetime import date, datetime, timedelta

import streamlit as st
import pandas as pd

from db import get_conn
from schema import init_db

# ============ Config ============

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

def build_day_options_with_dates(week_monday: date):
    """
    Devuelve una lista de (idx, label) para los 5 días lectivos de la semana dada.
    Ejemplo de label: 'Lunes 09/03/26'
    """
    dias = [week_monday + timedelta(days=i) for i in range(5)]
    options = []
    for i, d in enumerate(dias):
        label = f"{DIAS_ES[i]} {d.strftime('%d/%m/%y')}"
        options.append((i, label))
    return options

# ====== Aulas ======
def get_rooms():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM rooms ORDER BY name;")
        return cur.fetchall()

# ======================================
#   USUARIOS — PostgreSQL (completo + admin: borrar/cambiar rol)
# ======================================

def no_users_exist() -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users;")
        return cur.fetchone()[0] == 0

def get_user_by_email(email: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, email, role, status, password_hash
            FROM users
            WHERE email=%s
        """, (email.lower().strip(),))
        return cur.fetchone()

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

def list_users():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, email, role, status, password_hash
            FROM users
            ORDER BY role DESC, name
        """)
        return cur.fetchall()

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

def set_user_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET password_hash=%s
            WHERE id=%s
        """, (password_hash, user_id))
        conn.commit()

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

def delete_user(user_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
            conn.commit()
            return True, "Usuario eliminado correctamente."
        except Exception as ex:
            return False, f"No se pudo eliminar el usuario: {ex}"

def update_user_role(user_id: int, new_role: str):
    if new_role not in ("profesor", "admin"):
        return False, "Rol no válido."
    with get_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                UPDATE users
                SET role=%s
                WHERE id=%s
            """, (new_role, user_id))
            conn.commit()
            return True, "Rol actualizado correctamente."
        except Exception as ex:
            return False, f"No se pudo actualizar el rol: {ex}"

# ======================================
#   RESERVAS — PostgreSQL
# ======================================

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

def create_reservation(room_id, fecha, slot_index, reserved_by_id, notes=""):
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

def delete_reservation(reservation_id):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM reservations WHERE id=%s", (reservation_id,))
        conn.commit()

def delete_reservations_range(room_id: int, start_date: date, end_date: date) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM reservations
            WHERE room_id=%s AND fecha BETWEEN %s AND %s
        """, (room_id, start_date, end_date))
        deleted = cur.rowcount or 0
        conn.commit()
        return deleted

def get_all_reservations_by_user(user_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT r.id, r.room_id, r.fecha, r.slot_index, r.reserved_by_id, r.notes, r.created_at,
                   rooms.name AS room_name
            FROM reservations r
            JOIN rooms ON rooms.id = r.room_id
            WHERE r.reserved_by_id = %s
            ORDER BY r.fecha ASC, r.slot_index ASC
        """, (user_id,))
        return cur.fetchall()

# ======================================
#   CUADRANTE / DATAFRAME — PostgreSQL
# ======================================

def build_availability_df(room_id, monday: date):
    dias = [monday + timedelta(days=i) for i in range(5)]
    reservas = list_reservations(room_id, dias[0], dias[-1])
    users_all = list_users()
    users_map = {row[0]: row[1] for row in users_all}  # id -> name

    booked = {}
    for (res_id, room_id, fecha, slot_index, reserved_by_id, notes, created_at) in reservas:
        booked[(fecha, slot_index)] = (reserved_by_id, notes)

    idx = [f"{s}-{e}" for s, e in SLOTS]
    cols = [f"{DIAS_ES[i]}\n{dias[i].strftime('%d/%m')}" for i in range(5)]

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

def df_semantico_a_plano(df_sem):
    df2 = df_sem.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(lambda x: x[1] if x[0] == "RESERVADO" else "Libre")
    return df2

# ======================================
#   IMPORTACIÓN DE PROFESORES — PostgreSQL
# ======================================

def import_profesores_from_excel(file) -> tuple[int, int, list]:
    try:
        df = pd.read_excel(file, engine="openpyxl")
    except Exception as ex:
        return 0, 0, [f"No se pudo leer el Excel: {ex}"]

    cols = {c.strip().lower(): c for c in df.columns}
    if "nombre" not in cols or "email" not in cols:
        return 0, 0, ["El Excel debe contener columnas 'Nombre' y 'Email'."]

    df = df.rename(columns={cols["nombre"]: "Nombre", cols["email"]: "Email"})
    df = df[["Nombre", "Email"]].copy()

    creados = 0
    actualizados = 0
    errores = []
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    with get_conn() as conn:
        cur = conn.cursor()
        for idx, row in df.iterrows():
            name = str(row["Nombre"]).strip()
            email = str(row["Email"]).strip().lower()
            if not name or not email:
                errores.append(f"Fila {idx+2}: nombre o email vacío.")
                continue
            if not email_re.match(email):
                errores.append(f"Fila {idx+2}: email inválido '{email}'.")
                continue

            cur.execute("SELECT id, role FROM users WHERE email=%s", (email,))
            existing = cur.fetchone()

            if existing:
                uid, role = existing
                if role == "profesor":
                    cur.execute("""
                        UPDATE users
                        SET name=%s, status='activo'
                        WHERE id=%s
                    """, (name, uid))
                    actualizados += 1
                else:
                    actualizados += 1
            else:
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
    df = pd.DataFrame([{"Nombre": "Nombre Apellido", "Email": "nombre.apellido@centro.es"}])
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Profesores")
    out.seek(0)
    return out.read(), "profesores_template.xlsx"

# ======================================
#   EXPORTACIONES — EXCEL y PDF
# ======================================

try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

def export_week_to_excel_bytes(df_sem, room_name: str, week_monday: date):
    df2 = df_semantico_a_plano(df_sem)
    out = BytesIO()
    fname = f"cuadrante_{room_name}_{week_monday}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df2.to_excel(writer, index=True, sheet_name="Cuadrante")
        ws = writer.book["Cuadrante"]
        ws.freeze_panes = "B2"
        for col in ws.columns:
            try:
                max_len = max(len(str(c.value)) if c.value else 0 for c in col)
                col_letter = col[0].column_letter
                ws.column_dimensions[col_letter].width = min(max_len + 2, 30)
            except Exception:
                pass
    out.seek(0)
    return out.read(), fname

def export_week_to_pdf_bytes(df_sem, room_name: str, week_monday: date):
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab no está instalado en este entorno.")
    df2 = df_semantico_a_plano(df_sem)
    buf = BytesIO()
    fname = f"cuadrante_{room_name}_{week_monday}.pdf"
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter))
    headers = ["Hora"] + list(df2.columns)
    data = [headers]
    for idx, row in df2.iterrows():
        data.append([idx] + list(row))
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    doc.build([table])
    return buf.getvalue(), fname

# ======================================
#   ESTADÍSTICAS DE USO — PostgreSQL
# ======================================

def obtener_estadisticas():
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM reservations", conn)
    if df.empty:
        return None

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.weekday
    map_dias = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    df["dia_nombre"] = df["dia_semana"].map(map_dias)

    with get_conn() as conn:
        salas = pd.read_sql_query("SELECT id, name FROM rooms", conn)
    aulas_map = dict(zip(salas["id"], salas["name"]))
    df["aula_nombre"] = df["room_id"].map(aulas_map)

    with get_conn() as conn:
        users_df = pd.read_sql_query("SELECT id AS uid, name AS uname FROM users", conn)
    df = df.merge(users_df, left_on="reserved_by_id", right_on="uid", how="left")

    reservas_por_aula = df.groupby("aula_nombre")["id"].count().sort_values(ascending=False)
    reservas_por_profesor = df.groupby("uname")["id"].count().sort_values(ascending=False)
    reservas_por_dia = df[df["dia_semana"] <= 4].groupby("dia_nombre")["id"].count().reindex(
        ["Lunes","Martes","Miércoles","Jueves","Viernes"]
    )

    map_slots = {i: f"{SLOTS[i][0]}–{SLOTS[i][1]}" for i in range(len(SLOTS))}
    df["slot_label"] = df["slot_index"].map(map_slots)
    reservas_por_franja = df.groupby("slot_label")["id"].count().reindex(list(map_slots.values()))

    return {
        "raw": df,
        "por_aula": reservas_por_aula,
        "por_profesor": reservas_por_profesor,
        "por_dia": reservas_por_dia,
        "por_franja": reservas_por_franja,
    }

# ======================================
#   AUTENTICACIÓN STREAMLIT
# ======================================

def login_screen():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)

    # LOGO (centrado)
    try:
        st.image("logo.png", width=140)
    except:
        st.write("")

    # TÍTULO
    st.markdown("""
        <h2 style="margin-top: 10px; margin-bottom: 2px;">Reserva de aulas</h2>
        <h4 style="color: #333; font-weight: normal; margin-top: 0px;">
            IES Antonio García Bellido
        </h4>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Email y password en la MISMA pantalla
    email = st.text_input("Email institucional", key="login_email")
    password = st.text_input("Contraseña", type="password", key="login_password")

    login_btn = st.button("Entrar", key="login_btn")

    if login_btn:
        u = get_user_by_email(email)
        if not u:
            st.error("Email no registrado.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        uid, name, email, role, status, pwd_hash = u

        if role == "profesor" and status != "activo":
            st.error("Tu cuenta está suspendida. Contacta con un administrador.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Primer acceso sin contraseña
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

        # Falta contraseña
        if not password:
            st.error("Introduce la contraseña.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # Login correcto
        if verify_password(password, pwd_hash):
            st.session_state["user"] = {
                "id": uid,
                "name": name,
                "email": email,
                "role": role,
                "status": status
            }
            st.markdown("</div>", unsafe_allow_html=True)
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")

    st.markdown("</div>", unsafe_allow_html=True)

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
                "id": uid, "name": name, "email": email, "role": role, "status": status
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
#   NAVEGACIÓN SEMANAL + SELECTORES + CUADRANTE
# ======================================

def render_week_navigation():
    if "week_monday" not in st.session_state:
        st.session_state["week_monday"] = lunes_de_semana(date.today())

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ Semana anterior", key="prev_week"):
            st.session_state["week_monday"] -= timedelta(days=7)
    with c3:
        if st.button("➡️ Semana siguiente", key="next_week"):
            st.session_state["week_monday"] += timedelta(days=7)

    week_start = st.session_state["week_monday"]
    week_end = week_start + timedelta(days=4)
    label = f"Semana del {week_start.strftime('%d/%m/%Y')} al {week_end.strftime('%d/%m/%Y')}"

    with c2:
        sel_date = st.date_input(label, value=week_start, key="go_to_week")
        st.session_state["week_monday"] = lunes_de_semana(sel_date)

    return st.session_state["week_monday"]

def render_room_selector():
    rooms = get_rooms()
    room_map = {rid: name for rid, name in rooms}
    options = [None] + list(room_map.keys())
    room_id = st.selectbox(
        "Aula",
        options,
        index=0,
        format_func=lambda r: "" if r is None else room_map[r],
        key="room_select"
    )
    room_name = None if room_id is None else room_map[room_id]
    return room_id, room_name

def render_weekly_grid(room_id, room_name, week_monday):
    if room_id is None:
        st.info("Selecciona un aula para ver el cuadrante.")
        return None
    week_start = week_monday
    week_end = week_monday + timedelta(days=4)
    st.markdown(f"### 🗓 Semana del **{week_start.strftime('%d/%m/%Y')}** al **{week_end.strftime('%d/%m/%Y')}**")
    df_sem = build_availability_df(room_id, week_monday)
    df_disp = df_sem.copy()
    for col in df_disp.columns:
        df_disp[col] = df_disp[col].apply(lambda x: f"🟥 {x[1]}" if x[0] == "RESERVADO" else "🟩 Libre")
    st.subheader(f"Aula: {room_name}")
    st.dataframe(df_disp, use_container_width=True)
    return df_sem

# ======================================
#   RESERVAS (UI general) — profesor/admin (admin puede reservar en su nombre)
# ======================================

def _select_dia_placeholder(key):
    opciones = [None] + list(range(5))
    return st.selectbox("Día", opciones, index=0, format_func=lambda i: "" if i is None else DIAS_ES[i], key=key)

def _select_slot_placeholder(key):
    opciones = [None] + list(range(len(SLOTS)))
    return st.selectbox("Hora", opciones, index=0,
                        format_func=lambda i: "" if i is None else f"{SLOTS[i][0]}–{SLOTS[i][1]}",
                        key=key)

def _select_profesor_placeholder(key):
    profesores = list_profesores(include_suspended=False)
    pm = {pid: f"{n} ({e})" for pid, n, e in profesores}
    opciones = [None] + list(pm.keys())
    sel = st.selectbox("Profesor", opciones, index=0, format_func=lambda x: "" if x is None else pm[x], key=key)
    return sel

def render_create_reservation(usuario, room_id, week_monday):
    st.markdown("### ➕ Nueva reserva")
    if room_id is None:
        st.warning("Selecciona un aula arriba para poder reservar.")
        return

    # Admin: reservar en su nombre o para otro
    reserved_by_id = None
    if usuario["role"] == "profesor":
        reserved_by_id = usuario["id"]
        st.info(f"Profesor: **{usuario['name']}**")
    else:
        modo = st.radio("Reservar:", ["En mi nombre", "Para un profesor"], horizontal=True, key="admin_res_mode")
        if modo == "En mi nombre":
            reserved_by_id = usuario["id"]
            st.info(f"Reservando en nombre de **{usuario['name']}**")
        else:
            profesores = list_profesores(include_suspended=False)
            if not profesores:
                st.error("No hay profesores activos registrados.")
                return
            pm = {pid: f"{n} ({e})" for pid, n, e in profesores}
            reserved_by_id = st.selectbox(
                "Profesor",
                [None] + list(pm.keys()),
                index=0,
                format_func=lambda x: "" if x is None else pm[x],
                key="reserve_prof"
            )

    # Día con etiqueta "Día + fecha" y placeholder en blanco
    day_options = build_day_options_with_dates(week_monday)  # [(0,"Lunes 09/03/26"), ...]
    day_idx = st.selectbox(
        "Día",
        [None] + [opt[0] for opt in day_options],
        index=0,
        format_func=lambda i: "" if i is None else dict(day_options)[i],
        key="reserve_day"
    )

    # Hora con placeholder en blanco
    slot_idx = st.selectbox(
        "Hora",
        [None] + list(range(len(SLOTS))),
        index=0,
        format_func=lambda i: "" if i is None else f"{SLOTS[i][0]}–{SLOTS[i][1]}",
        key="reserve_slot"
    )

    notes = st.text_input("Notas", key="reserve_notes")

    disabled = any(v is None for v in (reserved_by_id, day_idx, slot_idx))
    if st.button("Reservar", key="btn_reservar", disabled=disabled):
        fecha = week_monday + timedelta(days=day_idx)
        hoy = date.today()

        # Reglas para profesor (admin sin límites)
        if usuario["role"] == "profesor":
            if fecha < hoy:
                st.error("Día pasado.")
                return
            if fecha == hoy:
                inicio = datetime.strptime(SLOTS[slot_idx][0], "%H:%M").time()
                if datetime.now().time() > inicio:
                    st.error("Esa franja ya pasó hoy.")
                    return
            if (fecha - hoy).days > 7:
                st.error("Máximo 7 días de antelación.")
                return

        ok, msg = create_reservation(room_id, fecha, slot_idx, reserved_by_id, notes)
        (st.success if ok else st.error)(msg)

def render_cancel_reservation(usuario, room_id, week_monday):
    # ============================
    # PROFESOR: ver TODAS SUS RESERVAS globales
    # ============================
    if usuario["role"] == "profesor":
        st.markdown("### 🧾 Tus reservas (todas las aulas y fechas)")

        reservas = get_all_reservations_by_user(usuario["id"])

        if not reservas:
            st.info("No tienes reservas para cancelar.")
            return

        opciones = []
        for r in reservas:
            rid, room_id_r, fecha, slot, by_id, notes_r, created_at, room_name = r
            etiqueta = (
                f"{fecha.strftime('%d/%m/%Y')} · "
                f"{room_name} · "
                f"{DIAS_ES[fecha.weekday()]} · "
                f"{SLOTS[slot][0]}–{SLOTS[slot][1]}"
            )
            if notes_r:
                etiqueta += f" · {notes_r}"
            opciones.append((rid, etiqueta))

        sel = st.selectbox(
            "Selecciona la reserva a cancelar:",
            [None] + opciones,
            index=0,
            format_func=lambda x: "" if x is None else x[1],
            key="prof_cancel_sel"
        )

        if st.button("Cancelar reserva", key="cancel_prof_btn", disabled=(sel is None)):
            delete_reservation(sel[0])
            st.success("Reserva cancelada.")
        return

    # ============================
    # ADMIN: ver reservas de ESTA AULA y ESTA SEMANA
    # ============================

    st.markdown("### 🧾 Cancelar reserva en esta semana y aula")

    if room_id is None:
        st.info("Selecciona un aula para ver/cancelar reservas.")
        return

    reservas = list_reservations(room_id, week_monday, week_monday + timedelta(days=4))
    users_all = list_users()
    users_map = {uid: name for (uid, name, email, role, status, pwh) in users_all}

    opciones = []
    for r in reservas:
        rid, _, fdate, slot, by_id, notes_r, _ = r
        etiqueta = (
            f"{fdate.strftime('%d/%m/%Y')} · "
            f"{DIAS_ES[fdate.weekday()]} · "
            f"{SLOTS[slot][0]}–{SLOTS[slot][1]} · "
            f"{users_map.get(by_id, '—')}"
        )
        if notes_r:
            etiqueta += f" · {notes_r}"
        opciones.append((rid, etiqueta))

    if not opciones:
        st.info("No hay reservas en esta semana para este aula.")
        return

    sel = st.selectbox(
        "Reserva",
        [None] + opciones,
        index=0,
        format_func=lambda x: "" if x is None else x[1],
        key="reserve_cancel_sel_admin"
    )

    if st.button("Cancelar", key="btn_cancelar_admin", disabled=(sel is None)):
        delete_reservation(sel[0])
        st.warning("Reserva cancelada.")

# ======================================
#   ADMIN — Export/Backup
# ======================================

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
    st.subheader("📤 Exportar cuadrante / Backup")
    c1, c2, c3 = st.columns(3)
    with c1:
        if df_sem is not None and room_name:
            try:
                xb, xf = export_week_to_excel_bytes(df_sem, room_name, week_monday)
                st.download_button("💾 Excel", xb, xf, key="export_excel")
            except Exception as ex:
                st.error(f"Error exportando a Excel: {ex}")
    with c2:
        if df_sem is not None and room_name:
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

# ======================================
#   ADMIN — Gestión de usuarios (con borrar/cambiar rol)
# ======================================

def render_admin_user_management(usuario):
    if usuario["role"] != "admin":
        return

    st.subheader("👥 Gestión de usuarios")

    st.markdown("**Importar profesores desde Excel**")
    cta1, cta2 = st.columns([1, 1])
    with cta1:
        template_bytes, template_name = download_profesores_template_bytes()
        st.download_button("📄 Descargar plantilla", template_bytes, template_name, key="dl_template")
    with cta2:
        up = st.file_uploader("Sube 'profesores.xlsx' (Nombre, Email)", type=["xlsx"], key="upload_prof")
        if up is not None:
            creados, actualizados, errores = import_profesores_from_excel(up)
            st.success(f"Profesores creados: {creados} · actualizados: {actualizados}")
            if errores:
                with st.expander("Ver detalles de errores"):
                    for e in errores:
                        st.write("• " + e)

    st.markdown("---")
    st.markdown("**Crear usuario manualmente**")
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
    st.dataframe(df_users.drop(columns=["PasswordHash"]), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Cambiar estado (activo/suspendido)**")
    all_users_map = {uid: f"{name} <{email}> [{role}] ({estado})" for uid, name, email, role, estado, pwh in users_raw}
    if all_users_map:
        uid_sel = st.selectbox("Usuario", [None] + list(all_users_map.keys()), index=0,
                               format_func=lambda k: "" if k is None else all_users_map[k],
                               key="status_user_sel")
        new_status = st.selectbox("Nuevo estado", ["activo", "suspendido"], key="status_new")
        if st.button("Actualizar estado", key="btn_update_status", disabled=(uid_sel is None)):
            ok, msg = set_user_status(uid_sel, new_status)
            (st.success if ok else st.error)(msg)

    st.markdown("---")
    st.markdown("**Resetear contraseña de profesor**")
    profs_all = [u for u in users_raw if u[3] == "profesor"]
    if profs_all:
        prof_map = {uid: f"{name} <{email}> ({estado})" for uid, name, email, role, estado, pwh in profs_all}
        psel = st.selectbox("Profesor", [None] + list(prof_map.keys()), index=0,
                            format_func=lambda k: "" if k is None else prof_map[k], key="reset_prof_sel")
        np1 = st.text_input("Nueva contraseña", type="password", key="reset_pw1")
        np2 = st.text_input("Repetir contraseña", type="password", key="reset_pw2")
        if st.button("Resetear contraseña", key="btn_reset_pw", disabled=(psel is None)):
            if np1 != np2:
                st.error("Las contraseñas no coinciden.")
            elif len(np1) < 4:
                st.error("Contraseña demasiado corta.")
            else:
                set_user_password(psel, hash_password_secure(np1))
                st.success("Contraseña reseteada correctamente.")
    else:
        st.info("Aún no hay profesores registrados.")

    st.markdown("---")
    st.markdown("**Borrar usuario**")
    delete_map = {uid: f"{name} <{email}> ({role})" for uid, name, email, role, estado, pwh in users_raw}
    uid_del = st.selectbox("Selecciona el usuario a eliminar", [None] + list(delete_map.keys()), index=0,
                           format_func=lambda k: "" if k is None else delete_map[k],
                           key="delete_user_select")
    if st.button("Borrar usuario", key="btn_delete_user", disabled=(uid_del is None)):
        if uid_del == usuario["id"]:
            st.error("No puedes borrarte a ti mismo.")
        else:
            ok, msg = delete_user(uid_del)
            (st.success if ok else st.error)(msg)

    st.markdown("---")
    st.markdown("**Cambiar rol (profesor ↔ admin)**")
    role_map = {uid: f"{name} <{email}> [{role}]" for uid, name, email, role, estado, pwh in users_raw}
    uid_role = st.selectbox("Usuario", [None] + list(role_map.keys()), index=0,
                            format_func=lambda k: "" if k is None else role_map[k],
                            key="role_user_select")
    new_role = st.selectbox("Nuevo rol", ["profesor", "admin"], key="set_role_select")
    if st.button("Actualizar rol", key="btn_update_role", disabled=(uid_role is None)):
        if uid_role == usuario["id"]:
            st.error("No puedes cambiar tu propio rol.")
        else:
            ok, msg = update_user_role(uid_role, new_role)
            (st.success if ok else st.error)(msg)

# ======================================
#   ADMIN — Reservas recurrentes
# ======================================

def render_recurring_reservations(usuario):
    if usuario["role"] != "admin":
        return

    st.subheader("📆 Reservas recurrentes")

    profesores = list_profesores(include_suspended=False)
    if not profesores:
        st.warning("No hay profesores activos. Importa o crea alguno primero.")
        return

    mp = {pid: f"{n} ({e})" for pid, n, e in profesores}
    pid = st.selectbox(
        "Profesor",
        [None] + list(mp.keys()),
        index=0,
        format_func=lambda x: "" if x is None else mp[x],
        key="rec_pid"
    )

    rm = {rid: name for rid, name in get_rooms()}
    room_rec = st.selectbox(
        "Aula",
        [None] + list(rm.keys()),
        index=0,
        format_func=lambda r: "" if r is None else rm[r],
        key="rec_room"
    )

    # Día semanal (solo día, sin fecha)
    day_idx_rec = st.selectbox(
        "Día semanal",
        [None] + list(range(5)),
        index=0,
        format_func=lambda i: "" if i is None else DIAS_ES[i],
        key="rec_day"
    )

    slot_idx_rec = st.selectbox(
        "Franja horaria",
        [None] + list(range(len(SLOTS))),
        index=0,
        format_func=lambda i: "" if i is None else f"{SLOTS[i][0]}–{SLOTS[i][1]}",
        key="rec_slot"
    )

    notes_rec = st.text_input("Notas", key="rec_notes")

    disabled = any(v is None for v in (pid, room_rec, day_idx_rec, slot_idx_rec))
    if st.button("Crear reservas recurrentes", key="rec_submit", disabled=disabled):
        hoy = date.today()
        fin = fin_de_curso(hoy)
        delta = (day_idx_rec - hoy.weekday()) % 7
        fecha = hoy + timedelta(days=delta)

        creadas = 0
        conflictos = 0
        while fecha <= fin:
            if (not profesor_tiene_reserva(fecha, slot_idx_rec, pid)) and \
               (not has_conflict(room_rec, fecha, slot_idx_rec)):
                ok, _ = create_reservation(room_rec, fecha, slot_idx_rec, pid, notes_rec)
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
#   ADMIN — Borrado masivo por rango
# ======================================

def render_admin_bulk_delete(usuario):
    if usuario["role"] != "admin":
        return
    st.subheader("🗑️ Cancelar todas las reservas de un aula en un rango")
    rm = {rid: name for rid, name in get_rooms()}
    room_id = st.selectbox("Aula", [None] + list(rm.keys()), index=0,
                           format_func=lambda r: "" if r is None else rm[r],
                           key="bulk_room")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", value=date.today(), key="bulk_start")
    with col2:
        end_date = st.date_input("Fecha fin", value=date.today() + timedelta(days=7), key="bulk_end")

    confirm = st.checkbox("Confirmo que deseo cancelar TODAS las reservas en el rango.", key="bulk_confirm")
    disabled = (room_id is None) or (start_date is None) or (end_date is None) or (end_date < start_date) or (not confirm)
    if st.button("Cancelar reservas del rango", key="bulk_delete", disabled=disabled):
        deleted = delete_reservations_range(room_id, start_date, end_date)
        st.warning(f"Se han cancelado {deleted} reservas en {rm.get(room_id)} entre {start_date.strftime('%d/%m/%Y')} y {end_date.strftime('%d/%m/%Y')}.")

  # ======================================
#   SIDEBAR
# ======================================

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

##################################
# ADMIN -- STATS
##################################
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


# ======================================
#   MAIN
# ======================================

def main():
    st.set_page_config(page_title="Reserva de Aulas", layout="wide")

    # Estilo global: fondo verde degradado + inputs mejorados
    st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(to bottom right, #f2fbf5, #e1f5e8);
    }
    /* Contenedor del login */
    .login-container {
        background-color: rgba(255,255,255,0.90);
        padding: 30px;
        border-radius: 12px;
        width: 420px;
        margin: auto;
        margin-top: 60px;
        text-align: center;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
    }
    /* Inputs */
    input[type="text"], input[type="password"] {
        border: 1px solid #666 !important;
        background-color: #f8f8f8 !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    init_db()

    # Flujo de autenticación
    if no_users_exist():
        bootstrap_admin_screen()
        return

    if "needs_password_setup" in st.session_state:
        first_password_screen()
        return

    if "ask_password" in st.session_state:
        password_login_screen()
        return

    # LOGIN (solo si no hay usuario en sesión)
    if "user" not in st.session_state:
        login_screen()
        return

    # Ya hay usuario en sesión
    usuario = st.session_state["user"]

    # Sidebar
    render_sidebar(usuario)

    # LOGO + TÍTULOS CENTRADOS
    st.markdown("""
    <div style="text-align: center; padding-top: 10px;">
        <img src="logo.png" width="120">
        <h1 style="margin-bottom: 0px;">Reserva de Aulas</h1>
        <h4 style="margin-top: 4px; color: #444;">IES Antonio García Bellido</h4>
    </div>
    """, unsafe_allow_html=True)

    # Navegación semanal
    week_monday = render_week_navigation()

    # Selector de aula
    room_id, room_name = render_room_selector()

    # Cuadrante
    df_sem = render_weekly_grid(room_id, room_name, week_monday)

    # PROFESOR → no pestañas
    if usuario["role"] != "admin":
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            render_create_reservation(usuario, room_id, week_monday)
        with col2:
            render_cancel_reservation(usuario, room_id, week_monday)
        return

    # ADMIN
    st.divider()
    tabs = st.tabs([
        "📌 Reservas",
        "📆 Recurrentes",
        "🗑️ Borrado por rango",
        "👥 Usuarios",
        "📤 Export/Backup",
        "📈 Estadísticas",
    ])

    with tabs[0]:
        st.subheader("📌 Reservas (admin)")
        col1, col2 = st.columns(2)
        with col1:
            render_create_reservation(usuario, room_id, week_monday)
        with col2:
            render_cancel_reservation(usuario, room_id, week_monday)

    with tabs[1]:
        render_recurring_reservations(usuario)

    with tabs[2]:
        render_admin_bulk_delete(usuario)

    with tabs[3]:
        render_admin_user_management(usuario)

    with tabs[4]:
        render_admin_exports_and_backup(usuario, df_sem, room_name, week_monday)

    with tabs[5]:
        render_admin_stats(usuario)

if __name__ == "__main__":
    main()
