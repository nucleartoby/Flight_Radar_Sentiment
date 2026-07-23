import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import Config


# Columns on flight_events that a tracker event dict is expected to carry
EVENT_COLUMNS = [
    "icao24", "callsign", "aircraft_type", "aircraft_name", "classification", "score",
    "first_seen_ts", "last_seen_ts", "duration_min", "start_reason", "end_reason",
    "origin_lat", "origin_lon", "origin_base", "origin_airport",
    "dest_lat", "dest_lon", "dest_base", "dest_airport",
    "n_points", "max_alt_m", "min_alt_m", "max_speed", "path_distance_km", "bases_visited",
    "bz_price_start", "cl_price_start", "bz_price_end", "cl_price_end",
    "status", "updated_ts",]


def connect(db_path: str = None) -> sqlite3.Connection:
    path = db_path or Config.DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS positions (
            icao24         TEXT    NOT NULL,
            ts             INTEGER NOT NULL,
            lat            REAL,
            lon            REAL,
            alt_m          REAL,
            speed          REAL,
            heading        REAL,
            on_ground      INTEGER,
            callsign       TEXT,
            classification TEXT,
            score          INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_positions_icao_ts ON positions(icao24, ts);

        CREATE TABLE IF NOT EXISTS flight_events (
            event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            icao24           TEXT    NOT NULL,
            callsign         TEXT,
            aircraft_type    TEXT,
            aircraft_name    TEXT,
            classification   TEXT,
            score            INTEGER,
            first_seen_ts    INTEGER,
            last_seen_ts     INTEGER,
            duration_min     REAL,
            start_reason     TEXT,
            end_reason       TEXT,
            origin_lat       REAL,
            origin_lon       REAL,
            origin_base      TEXT,
            origin_airport   TEXT,
            dest_lat         REAL,
            dest_lon         REAL,
            dest_base        TEXT,
            dest_airport     TEXT,
            n_points         INTEGER,
            max_alt_m        REAL,
            min_alt_m        REAL,
            max_speed        REAL,
            path_distance_km REAL,
            bases_visited    TEXT,
            bz_price_start   REAL,
            cl_price_start   REAL,
            bz_price_end     REAL,
            cl_price_end     REAL,
            status           TEXT    NOT NULL DEFAULT 'active',
            updated_ts       INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_events_status ON flight_events(status);
        CREATE INDEX IF NOT EXISTS idx_events_icao ON flight_events(icao24);

        CREATE TABLE IF NOT EXISTS oil_prices_intraday (
            symbol TEXT    NOT NULL,
            ts     INTEGER NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_oil_symbol_ts ON oil_prices_intraday(symbol, ts);
        """
    )
    conn.commit()


def insert_positions(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """INSERT INTO positions
           (icao24, ts, lat, lon, alt_m, speed, heading, on_ground, callsign, classification, score)
           VALUES (:icao24, :ts, :lat, :lon, :alt_m, :speed, :heading, :on_ground,
                   :callsign, :classification, :score)""",
        rows,)
    conn.commit()


def upsert_active_event(conn: sqlite3.Connection, event: Dict) -> int:
    payload = {c: event.get(c) for c in EVENT_COLUMNS}
    event_id = event.get("event_id")

    if event_id is None:
        cols = ", ".join(EVENT_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in EVENT_COLUMNS)
        cur = conn.execute(
            f"INSERT INTO flight_events ({cols}) VALUES ({placeholders})", payload
        )
        conn.commit()
        return cur.lastrowid

    assignments = ", ".join(f"{c} = :{c}" for c in EVENT_COLUMNS)
    payload["event_id"] = event_id
    conn.execute(
        f"UPDATE flight_events SET {assignments} WHERE event_id = :event_id", payload
    )
    conn.commit()
    return event_id


def close_event(conn: sqlite3.Connection, event: Dict) -> None:
    event["status"] = "closed"
    upsert_active_event(conn, event)


def load_active_events(conn: sqlite3.Connection) -> Dict[str, Dict]:
    rows = conn.execute(
        "SELECT * FROM flight_events WHERE status = 'active'").fetchall()
    return {row["icao24"]: dict(row) for row in rows}


def upsert_oil(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """INSERT INTO oil_prices_intraday (symbol, ts, open, high, low, close, volume)
           VALUES (:symbol, :ts, :open, :high, :low, :close, :volume)
           ON CONFLICT(symbol, ts) DO UPDATE SET
               open=excluded.open, high=excluded.high, low=excluded.low,
               close=excluded.close, volume=excluded.volume""",
        rows,)
    conn.commit()


def price_asof(conn: sqlite3.Connection, symbol: str, ts: int) -> Optional[float]:
    row = conn.execute(
        """SELECT close FROM oil_prices_intraday
           WHERE symbol = ? AND ts <= ? ORDER BY ts DESC LIMIT 1""",
        (symbol, ts),).fetchone()
    return row["close"] if row else None
