# app.py — aplicación completa en un solo archivo

import sqlite3
from datetime import date, datetime, timedelta
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
import hashlib

# =========================
# Dependencias opcionales PDF
# =========================
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

DB_PATH = Path("reservas.db")

# =========================
# Configuración básica
# =========================
SLOTS = [
    ("08:40", "09:30"),
    ("09:35", "10:25"),
    ("10:30", "11:20"),
    ("11:50", "12:40"),
    ("12:45", "13:35"),
    ("13:40", "14:30"),
]
DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

# =========================
# Utilidades
# =========================
def lunes_de_semana(d: date) -> date:
    return d - timedelta(days=d.weekday())

def fechas_semana(lunes: date):
    return [lunes + timedelta(days=i) for i in range(5)]

def fin_de_curso(hoy: date) -> date:
    if hoy.month >= 7:
        return date(hoy.year + 1, 6, 30)
    return date(hoy.year, 6, 30)

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def get_conn():
    return sqlite3.connect(DB_PATH)

# =========================
# Base de datos
# =========================
def init_db():
    with get_conn() as conn:
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL CHECK(role IN ('profesor','admin')),
            password_hash TEXT
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL,
            fecha TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            reserved_by TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(room_id, fecha, slot_index)
        )
        """)

        conn.commit()

        # Aulas base
        aulas = [
            ("Informática A (206)",),
            ("Informática B (210)",),
            ("Informática C (209)",),
            ("Biblioteca",),
        ]
        for a in aulas:
            c.execute("INSERT OR IGNORE INTO rooms(name) VALUES(?)", a)

        conn.commit()

# --- Usuarios
def no_users_exist() -> bool:
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return n == 0

def get_user_by_email(email: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id,name,email,role,password_hash FROM users WHERE email=?",
            (email.lower(),)
        ).fetchone()

def create_user(name: str, email: str, role: str):
    name = (name or "").strip()
    email = (email or "").strip().lower()
    role = (role or "").strip().lower()
    if not name or not email:
        return False, "Nombre y email obligatorios."
    if role not in ("profesor", "admin"):
        return False, "Rol no válido."
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users(name,email,role,password_hash) VALUES (?,?,?,NULL)",
                (name, email, role)
            )
            conn.commit()
            return True, "Usuario creado correctamente (primer acceso sin contraseña)."
        except sqlite3.IntegrityError:
            return False, "Ese email ya existe."

def list_users():
    with get_conn() as conn:
        return conn.execute("SELECT id,name,email,role,password_hash FROM users").fetchall()

def list_profesores():
    with get_conn() as conn:
        return conn.execute("SELECT id,name,email FROM users WHERE role='profesor' ORDER BY name").fetchall()

def set_user_password(user_id: int, password_hash: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, user_id))
        conn.commit()

# --- Aulas
def get_rooms():
    with get_conn() as conn:
        return conn.execute("SELECT id,name FROM rooms ORDER BY name").fetchall()

# --- Reservas
def list_reservations(room_id, start_date, end_date):
    with get_conn() as conn:
        return conn.execute("""
            SELECT id,room_id,fecha,slot_index,reserved_by,notes,created_at
            FROM reservations
            WHERE room_id=? AND fecha BETWEEN ? AND ?
        """, (room_id, start_date.isoformat(), end_date.isoformat())).fetchall()

def has_conflict(room_id, fecha, slot_index):
    with get_conn() as conn:
        r = conn.execute("""
            SELECT 1 FROM reservations
            WHERE room_id=? AND fecha=? AND slot_index=?
        """, (room_id, fecha.isoformat(), slot_index)).fetchone()
        return r is not None

def profesor_tiene_reserva(fecha, slot_index, profesor):
    """Comprueba si el profesor ya tiene una reserva en esa fecha y franja, independientemente del aula."""
    with get_conn() as conn:
        r = conn.execute("""
            SELECT 1 FROM reservations
            WHERE fecha=? AND slot_index=? AND reserved_by=?
        """, (fecha.isoformat(), slot_index, profesor)).fetchone()
        return r is not None

def create_reservation(room_id, fecha, slot_index, reserved_by, notes=""):
    if has_conflict(room_id, fecha, slot_index):
        return False, "Ya existe una reserva en esa franja para este aula."
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO reservations(room_id, fecha, slot_index, reserved_by, notes, created_at)
            VALUES (?,?,?,?,?,?)
        """, (room_id, fecha.isoformat(), slot_index, reserved_by, notes, datetime.now().isoformat()))
        conn.commit()
    return True, "Reserva creada."

def delete_reservation(res_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM reservations WHERE id=?", (res_id,))
        conn.commit()

# =========================
# Login / Primer acceso
# =========================
def login_screen():
    st.title("🔐 Acceso")
    email = st.text_input("Email institucional", key="login_email")

    if st.button("Continuar", key="login_continue"):
        u = get_user_by_email(email)
        if not u:
            st.error("Email no registrado.")
            return
        uid, name, email, role, pwd_hash = u
        if pwd_hash is None:
            st.session_state["pending_user"] = {"id": uid, "name": name, "email": email, "role": role}
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
        set_user_password(u["id"], hash_password(pwd1))
        st.success("Contraseña creada. Inicia sesión.")
        st.session_state.clear()
        st.rerun()

def password_login_screen():
    u = st.session_state["login_user"]
    uid, name, email, role, pwd_hash = u
    st.title("🔒 Introduce tu contraseña")
    pwd = st.text_input("Contraseña", type="password", key="pl_pass")

    if st.button("Entrar", key="pl_enter"):
        if hash_password(pwd) == pwd_hash:
            st.session_state["user"] = {"id": uid, "name": name, "email": email, "role": role}
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
        ok, msg = create_user(name, email, "admin")
        if ok:
            uid = get_user_by_email(email)[0]
            set_user_password(uid, hash_password(p1))
            st.success("Administrador creado. Inicia sesión.")
            st.session_state.clear()
            st.rerun()
        else:
            st.error(msg)

# =========================
# Exportaciones
# =========================
def df_semantico_a_plano(df_sem):
    df2 = df_sem.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(lambda x: x[1] if x[0] == "RESERVADO" else "Libre")
    return df2

def export_week_to_excel_bytes(df_sem, room_name: str, week_monday: date):
    df2 = df_semantico_a_plano(df_sem)
    out = BytesIO()
    fname = f"cuadrante_{room_name}_{week_monday}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df2.to_excel(writer, index=True, sheet_name="Cuadrante")
    out.seek(0)
    return out.read(), fname

def export_week_to_pdf_bytes(df_sem, room_name: str, week_monday: date):
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab no está instalado.")
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

# =========================
# Cuadrante (DataFrame semántico)
# =========================
def build_availability_df(room_id, monday):
    dias = fechas_semana(monday)
    reservas = list_reservations(room_id, dias[0], dias[-1])
    booked = {(r[2], r[3]): r for r in reservas}  # (fecha_iso, slot_idx) -> fila

    data = []
    idx = [f"{s}-{e}" for s, e in SLOTS]
    cols = [f"{DIAS_ES[i]}\n{dias[i].strftime('%d/%m')}" for i in range(5)]

    for slot in range(len(SLOTS)):
        fila = []
        for d in dias:
            key = (d.isoformat(), slot)
            if key in booked:
                fila.append(("RESERVADO", booked[key][4]))  # nombre
            else:
                fila.append(("LIBRE", ""))
        data.append(fila)

    return pd.DataFrame(data, index=idx, columns=cols)

# =========================
# ESTADÍSTICAS DE USO (ADMIN)
# =========================
def obtener_estadisticas():
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM reservations", conn)

    if df.empty:
        return None

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dia_semana"] = df["fecha"].dt.weekday  # 0=lunes

    map_dias = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    df["dia_nombre"] = df["dia_semana"].map(map_dias)

    aulas = {rid: name for rid, name in get_rooms()}
    df["aula_nombre"] = df["room_id"].map(aulas)

    reservas_por_aula = df.groupby("aula_nombre")["id"].count().sort_values(ascending=False)

    reservas_por_profesor = df.groupby("reserved_by")["id"].count().sort_values(ascending=False)

    # Sólo días lectivos en orden
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

# =========================
# APP
# =========================
def main():
    st.set_page_config(page_title="Reserva de Aulas", layout="wide")
    init_db()

    # Bootstrap si no hay usuarios
    if no_users_exist():
        bootstrap_admin_screen()
        return

    # Flujo de autenticación
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

    # Sidebar (logo + usuario + cambiar contraseña)
    with st.sidebar:
        st.image("logo.png", width=150)
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
            if hash_password(old) != u[4]:
                st.error("❌ La contraseña actual no coincide.")
            elif new1 != new2:
                st.error("❌ Las nuevas contraseñas no coinciden.")
            elif len(new1) < 4:
                st.error("❌ Contraseña demasiado corta.")
            else:
                set_user_password(usuario["id"], hash_password(new1))
                st.success("✔ Contraseña actualizada.")

        st.markdown("---")

    # Cabecera
    st.title("📚 Reserva de Aulas — IES Antonio García Bellido")

    # Navegación de semanas
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
        sel_date = st.date_input("Ir a semana:", value=st.session_state["week_monday"], key="go_to_week")
        st.session_state["week_monday"] = lunes_de_semana(sel_date)

    week_monday = st.session_state["week_monday"]
    week_start = week_monday
    week_end = week_monday + timedelta(days=4)
    st.markdown(f"### 🗓 Semana del **{week_start.strftime('%d/%m')}** al **{week_end.strftime('%d/%m')}**")

    # Aula
    rooms = get_rooms()
    room_map = {rid: name for rid, name in rooms}
    room_id = st.selectbox("Aula", list(room_map.keys()), format_func=lambda r: room_map[r], key="room_select")

    # Cuadrante
    df_sem = build_availability_df(room_id, week_monday)
    df_disp = df_sem.copy()
    for col in df_disp.columns:
        df_disp[col] = df_disp[col].apply(lambda x: f"🟥 {x[1]}" if x[0] == "RESERVADO" else "🟩 Libre")

    st.subheader(f"Aula: {room_map[room_id]}")
    st.dataframe(df_disp, use_container_width=True)

    # Exportar + Backup (admin)
    if usuario["role"] == "admin":
        st.markdown("### 📤 Exportar cuadrante")
        c1, c2, c3 = st.columns(3)
        with c1:
            try:
                xb, xf = export_week_to_excel_bytes(df_sem, room_map[room_id], week_monday)
                st.download_button("💾 Excel", xb, xf, key="export_excel")
            except Exception as ex:
                st.error(f"Error exportando a Excel: {ex}")
        with c2:
            try:
                pb, pf = export_week_to_pdf_bytes(df_sem, room_map[room_id], week_monday)
                st.download_button("🖨 PDF", pb, pf, key="export_pdf")
            except Exception as ex:
                st.warning("Para PDF instala reportlab: pip install reportlab")
                st.caption(f"Detalle: {ex}")
        with c3:
            # Backup de la BD
            try:
                with open(DB_PATH, "rb") as dbf:
                    st.download_button(
                        "🔐 Backup BD",
                        data=dbf,
                        file_name=f"backup_reservas_{date.today().isoformat()}.db",
                        mime="application/octet-stream",
                        key="backup_db"
                    )
            except Exception as ex:
                st.error(f"No se pudo leer la base de datos: {ex}")

    st.divider()
    col1, col2 = st.columns(2)

    # Crear reserva
    with col1:
        st.markdown("### ➕ Nueva reserva")
        if usuario["role"] == "profesor":
            reserved_by = usuario["name"]
            st.info(f"Profesor: **{reserved_by}**")
        else:
            profesores = list_profesores()
            if not profesores:
                st.error("No hay profesores registrados.")
                reserved_by = None
            else:
                pm = {pid: f"{n} ({e})" for pid, n, e in profesores}
                pid = st.selectbox("Profesor", list(pm.keys()), format_func=lambda x: pm[x], key="reserve_prof")
                prof_match = [n for (i, n, e) in profesores if i == pid]
                reserved_by = prof_match[0] if prof_match else None
                if reserved_by is None:
                    st.error("Profesor no encontrado.")

        day_idx = st.selectbox("Día", range(5), format_func=lambda i: DIAS_ES[i], key="reserve_day")
        slot_idx = st.selectbox("Hora", range(len(SLOTS)), format_func=lambda i: f"{SLOTS[i][0]}–{SLOTS[i][1]}", key="reserve_slot")
        notes = st.text_input("Notas", key="reserve_notes")

        if st.button("Reservar", key="btn_reservar"):
            if reserved_by is None:
                st.error("Selecciona un profesor válido.")
                st.stop()

            fecha = week_monday + timedelta(days=day_idx)
            hoy = date.today()

            if usuario["role"] == "profesor":
                if fecha < hoy:
                    st.error("Día pasado.")
                    st.stop()
                if fecha == hoy:
                    inicio = datetime.strptime(SLOTS[slot_idx][0], "%H:%M").time()
                    if datetime.now().time() > inicio:
                        st.error("Esa franja ya pasó hoy.")
                        st.stop()
                if (fecha - hoy).days > 7:
                    st.error("Máximo 7 días de antelación.")
                    st.stop()

            # Evitar reservas simultáneas del mismo profesor (cualquier aula)
            if profesor_tiene_reserva(fecha, slot_idx, reserved_by):
                st.error("Ese profesor ya tiene una reserva en esa franja horaria.")
                st.stop()

            ok, msg = create_reservation(room_id, fecha, slot_idx, reserved_by, notes)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    # Cancelar reserva
    with col2:
        st.markdown("### 🧾 Cancelar reserva")
        reservas = list_reservations(room_id, week_monday, week_monday + timedelta(days=4))
        opciones = []
        for r in reservas:
            rid, _, fstr, slot, by, notes_r, _ = r
            f = datetime.fromisoformat(fstr)
            etiqueta = f"{f.strftime('%d/%m')} · {DIAS_ES[f.weekday()]} · {SLOTS[slot][0]}–{SLOTS[slot][1]} · {by}"
            if notes_r:
                etiqueta += f" · {notes_r}"
            if usuario["role"] == "profesor" and by != usuario["name"]:
                continue
            opciones.append((rid, etiqueta))

        if not opciones:
            st.info("No tienes reservas que puedas cancelar.")
        else:
            sel = st.selectbox("Reserva", opciones, format_func=lambda x: x[1], key="reserve_cancel_sel")
            if st.button("Cancelar", key="btn_cancelar"):
                delete_reservation(sel[0])
                st.warning("Reserva cancelada.")
                st.rerun()

    # Reservas recurrentes (admin)
    if usuario["role"] == "admin":
        st.divider()
        st.header("📆 Reservas recurrentes")

    # Inicializa variables para evitar UnboundLocalError
    sub_rec = False
    reserved_rec = None
    room_rec = None
    day_idx_rec = None
    slot_idx_rec = None
    notes_rec = ""

    with st.form("form_rec"):
        profesores = list_profesores()
        if not profesores:
            st.warning("No hay profesores registrados. Crea alguno primero en 'Gestión de usuarios'.")
            # Botón deshabilitado para mantener UI coherente
            st.form_submit_button("Crear reservas recurrentes", disabled=True)
        else:
            mp = {pid: f"{n} ({e})" for pid, n, e in profesores}
            pid = st.selectbox("Profesor", list(mp.keys()), format_func=lambda x: mp[x], key="rec_pid")

            # Buscar el nombre del profesor con seguridad
            match_rec = [n for (i, n, e) in profesores if i == pid]
            if not match_rec:
                st.error("❌ Error interno: profesor no encontrado.")
                st.form_submit_button("Crear reservas recurrentes", disabled=True)
            else:
                reserved_rec = match_rec[0]

                rm = {rid: name for rid, name in get_rooms()}
                if not rm:
                    st.error("No hay aulas registradas.")
                    st.form_submit_button("Crear reservas recurrentes", disabled=True)
                else:
                    room_rec = st.selectbox("Aula", list(rm.keys()), format_func=lambda r: rm[r], key="rec_room")

                    day_idx_rec = st.selectbox("Día semanal", range(5), format_func=lambda i: DIAS_ES[i], key="rec_day")
                    slot_idx_rec = st.selectbox("Franja horaria", range(len(SLOTS)),
                                                format_func=lambda i: f"{SLOTS[i][0]}–{SLOTS[i][1]}",
                                                key="rec_slot")

                    notes_rec = st.text_input("Notas", key="rec_notes")

                    # Aquí sí se define sub_rec
                    sub_rec = st.form_submit_button("Crear reservas recurrentes", key="rec_submit")

    # Solo procesar si sub_rec es True y todo está definido
    if sub_rec and (reserved_rec is not None) and (room_rec is not None) \
       and (day_idx_rec is not None) and (slot_idx_rec is not None):

        hoy = date.today()
        fin = fin_de_curso(hoy)

        # Próxima ocurrencia del día elegido (incluye hoy si coincide)
        delta = (day_idx_rec - hoy.weekday()) % 7
        fecha = hoy + timedelta(days=delta)

        creadas = 0
        conflictos = 0

        while fecha <= fin:
            # Evitar simultánea del profesor y conflicto del aula
            if (not profesor_tiene_reserva(fecha, slot_idx_rec, reserved_rec)) and \
               (not has_conflict(room_rec, fecha, slot_idx_rec)):
                create_reservation(room_rec, fecha, slot_idx_rec, reserved_rec, notes_rec)
                creadas += 1
            else:
                conflictos += 1

            fecha += timedelta(days=7)

        st.success(f"✔ {creadas} reservas creadas.")
        if conflictos:
            st.warning(f"⚠ {conflictos} conflictos omitidos.")

    # Gestión de usuarios (admin)
    if usuario["role"] == "admin":
        st.divider()
        st.header("👥 Gestión de usuarios")
        with st.form("form_new_user", clear_on_submit=True):
            colA, colB, colC = st.columns([2, 2, 1])
            with colA:
                n = st.text_input("Nombre completo", key="new_user_name")
            with colB:
                e = st.text_input("Email", key="new_user_email")
            with colC:
                r = st.selectbox("Rol", ["profesor", "admin"], key="new_user_role")
            if st.form_submit_button("Crear usuario", key="new_user_btn"):
                ok, msg = create_user(n, e, r)
                (st.success if ok else st.error)(msg)

        st.dataframe(pd.DataFrame(
            list_users(),
            columns=["ID", "Nombre", "Email", "Rol", "PasswordHash"]
        ), hide_index=True)

    # Estadísticas (admin)
    if usuario["role"] == "admin":
        st.divider()
        st.header("📈 Estadísticas de uso")

        est = obtener_estadisticas()
        if est is None:
            st.info("Todavía no hay reservas registradas para generar estadísticas.")
        else:
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

# Entry point
if __name__ == "__main__":

    main()

