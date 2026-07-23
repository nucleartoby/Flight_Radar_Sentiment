import pandas as pd

from src.tracking import db


def main():
    conn = db.connect()

    def scalar(sql, params=()):
        return conn.execute(sql, params).fetchone()[0]

    n_positions = scalar("SELECT COUNT(*) FROM positions")
    n_aircraft = scalar("SELECT COUNT(DISTINCT icao24) FROM positions")
    n_active = scalar("SELECT COUNT(*) FROM flight_events WHERE status = ?", ("active",))
    n_closed = scalar("SELECT COUNT(*) FROM flight_events WHERE status = ?", ("closed",))
    n_oil = scalar("SELECT COUNT(*) FROM oil_prices_intraday")

    print("=" * 70)
    print("  FLIGHT TRACKING DB SUMMARY")
    print("=" * 70)
    print(f"positions rows        : {n_positions}")
    print(f"distinct aircraft     : {n_aircraft}")
    print(f"events (active)       : {n_active}")
    print(f"events (closed)       : {n_closed}")
    print(f"oil intraday rows     : {n_oil}")

    oil = pd.read_sql_query(
        "SELECT symbol, COUNT(*) n, MAX(ts) last_ts FROM oil_prices_intraday GROUP BY symbol",
        conn,)
    
    if not oil.empty:
        oil["last_utc"] = pd.to_datetime(oil["last_ts"], unit="s", utc=True)
        print("\n  Oil coverage:")
        print(oil[["symbol", "n", "last_utc"]].to_string(index=False))

    events = pd.read_sql_query(
        """SELECT icao24, callsign, aircraft_name, classification, status,
                  duration_min, path_distance_km, n_points, start_reason, end_reason,
                  origin_base, dest_base, bz_price_start, bz_price_end
           FROM flight_events ORDER BY event_id DESC LIMIT 10""",
        conn,)
    
    if not events.empty:
        print("\n  Most recent events:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(events.to_string(index=False))
    else:
        print("\n  No flight events yet.")

    conn.close()


if __name__ == "__main__":
    main()
