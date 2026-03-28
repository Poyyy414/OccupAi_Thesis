"""
database.py — Aiven PostgreSQL Connection for OccupAI
======================================================
.env file required in project root:
  DATABASE_URL=postgres://avnadmin:PASSWORD@host:port/defaultdb?sslmode=require
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_db():
    """Returns a new Aiven PostgreSQL connection using RealDictCursor
    so rows are accessible as dicts: row['email'] not row[0]"""
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not set in .env")
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            sslmode="require",
            cursor_factory=RealDictCursor
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        raise


def init_db():
    """Creates all tables. Run once: python database.py"""
    conn = get_db()
    cur  = conn.cursor()

    # ── users (parent) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       SERIAL PRIMARY KEY,
            first_name    VARCHAR(50)  NOT NULL,
            last_name     VARCHAR(50)  NOT NULL,
            full_name     VARCHAR(101)
                GENERATED ALWAYS AS (first_name || ' ' || last_name) STORED,
            email         VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role          VARCHAR(20)  NOT NULL DEFAULT 'driver'
                CHECK (role IN ('driver', 'admin', 'owner')),
            phone         VARCHAR(20)  DEFAULT '',
            is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
            created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login    TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

    # ── drivers (child) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id      SERIAL PRIMARY KEY,
            user_id        INTEGER UNIQUE NOT NULL
                REFERENCES users(user_id) ON DELETE CASCADE,
            license_number VARCHAR(50) DEFAULT '',
            vehicle_plate  VARCHAR(20) DEFAULT '',
            vehicle_type   VARCHAR(50) DEFAULT ''
        );
    """)

    # ── admins (child) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            admin_id     SERIAL PRIMARY KEY,
            user_id      INTEGER UNIQUE NOT NULL
                REFERENCES users(user_id) ON DELETE CASCADE,
            department   VARCHAR(100) DEFAULT '',
            access_level VARCHAR(20)  DEFAULT 'full'
                CHECK (access_level IN ('full', 'read_only'))
        );
    """)

    # ── owners (child) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS owners (
            owner_id      SERIAL PRIMARY KEY,
            user_id       INTEGER UNIQUE NOT NULL
                REFERENCES users(user_id) ON DELETE CASCADE,
            business_name VARCHAR(150) DEFAULT '',
            lot_address   VARCHAR(255) DEFAULT ''
        );
    """)

    # ── parking_logs ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS parking_logs (
            log_id        SERIAL PRIMARY KEY,
            occupied      INTEGER NOT NULL DEFAULT 0,
            free          INTEGER NOT NULL DEFAULT 0,
            total         INTEGER NOT NULL DEFAULT 0,
            occupancy_pct FLOAT   NOT NULL DEFAULT 0.0,
            lot_full      BOOLEAN NOT NULL DEFAULT FALSE,
            logged_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_parking_logs_time
        ON parking_logs(logged_at DESC);
    """)

    # ── slot_config (one row only) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS slot_config (
            slot_id    SERIAL PRIMARY KEY,
            slots      JSONB     NOT NULL DEFAULT '[]',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS one_row_slot_config
        ON slot_config((true));
    """)
    cur.execute("""
        INSERT INTO slot_config (slots) VALUES ('[]')
        ON CONFLICT DO NOTHING;
    """)

    # ── refresh_tokens ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL
                REFERENCES users(user_id) ON DELETE CASCADE,
            token      TEXT      NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ All tables created successfully!")


def test_connection():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        cur.close(); conn.close()
        print(f"✅ Connected to Aiven PostgreSQL!")
        print(f"   {dict(version)}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 OccupAI — Database Setup")
    print("=" * 40)
    if test_connection():
        print()
        init_db()
        print()
        print("✅ Done! Run: python app.py")
    else:
        print("❌ Fix your .env DATABASE_URL and try again.")