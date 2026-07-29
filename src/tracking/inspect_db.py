import sqlite3

import pandas as pd

from src.tracking import db


def main():
    conn = db.connect()

    def scalar(sql, params=(), default=0):
        try:
            return conn.execute(sql, params).fetchone()[0]
        except sqlite3.OperationalError:
            return default

    n_positions = scalar("SELECT COUNT(*) FROM positions")
    n_aircraft = scalar("SELECT COUNT(DISTINCT icao24) FROM positions")
    n_active = scalar("SELECT COUNT(*) FROM flight_events WHERE status = ?", ("active",))
    n_closed = scalar("SELECT COUNT(*) FROM flight_events WHERE status = ?", ("closed",))
    n_oil = scalar("SELECT COUNT(*) FROM oil_prices_intraday")
    n_rejected = scalar("SELECT COUNT(DISTINCT icao24) FROM rejected_mil_hex")

    print("=" * 70)
    print("  FLIGHT TRACKING DB SUMMARY")
    print("=" * 70)
    print(f"positions rows        : {n_positions}")
    print(f"distinct aircraft     : {n_aircraft}")
    print(f"events (active)       : {n_active}")
    print(f"events (closed)       : {n_closed}")
    print(f"oil intraday rows     : {n_oil}")
    print(f"rejected w/ mil hex   : {n_rejected}   (false-negative candidates)")

    has_ruleset = "ruleset_version" in {
        r[1] for r in conn.execute("PRAGMA table_info(flight_events)")}

    versions = pd.read_sql_query(
        """SELECT COALESCE(ruleset_version, 1) v, COUNT(*) n,
                  MIN(first_seen_ts) lo, MAX(first_seen_ts) hi
           FROM flight_events GROUP BY v ORDER BY v""", conn) if has_ruleset \
        else pd.DataFrame()
    if len(versions) > 1:
        print("\n  WARNING: events span multiple rulesets - scores are not comparable:")
        for _, r in versions.iterrows():
            lo = pd.to_datetime(r["lo"], unit="s", utc=True)
            hi = pd.to_datetime(r["hi"], unit="s", utc=True)
            print(f"    ruleset v{int(r['v'])}: {int(r['n'])} events  {lo} -> {hi}")

    oil = pd.read_sql_query(
        "SELECT symbol, COUNT(*) n, MAX(ts) last_ts FROM oil_prices_intraday GROUP BY symbol",
        conn,)
    
    if not oil.empty:
        oil["last_utc"] = pd.to_datetime(oil["last_ts"], unit="s", utc=True)
        print("\n  Oil coverage:")
        print(oil[["symbol", "n", "last_utc"]].to_string(index=False))

    hexcol = "mil_hex_block," if has_ruleset else ""
    events = pd.read_sql_query(
        f"""SELECT icao24, callsign, aircraft_name, classification, score,
                   {hexcol} status, duration_min, path_distance_km, n_points,
                   start_reason, end_reason, origin_base, dest_base,
                   bz_price_start, bz_price_end
            FROM flight_events ORDER BY event_id DESC LIMIT 10""",
        conn,)

    if not events.empty:
        print("\n  Most recent events:")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(events.to_string(index=False))

        evidence = pd.read_sql_query(
            """SELECT icao24, callsign, score, evidence_flags
               FROM flight_events
               WHERE evidence_flags IS NOT NULL
               ORDER BY event_id DESC LIMIT 5""", conn) if has_ruleset \
            else pd.DataFrame()
        if not evidence.empty:
            print("\n  Scoring evidence (most recent):")
            for _, r in evidence.iterrows():
                print(f"    {r['icao24']} {r['callsign'] or '-':<9} {r['score']:>3}  "
                      f"{r['evidence_flags']}")
    else:
        print("\n  No flight events yet.")

    conn.close()


if __name__ == "__main__":
    main()
