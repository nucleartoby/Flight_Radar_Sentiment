import logging
from typing import Dict, List, Optional

from geopy.distance import geodesic

from config.settings import Config
from src.tracking import db

# symbol - flight events column prefix
_OIL_COLS = {"BZ=F": "bz", "CL=F": "cl"}


class FlightTracker:
    def __init__(self, conn):
        self.conn = conn
        self.logger = logging.getLogger("tracker")
        self.active: Dict[str, Dict] = db.load_active_events(conn)
        self.last_pos: Dict[str, tuple] = {}       # icao24 lat lon
        self.last_on_ground: Dict[str, bool] = {}  # icao24  on ground
        if self.active:
            self.logger.info(f"Resumed {len(self.active)} active flight(s) from DB.")


    def update(self, snapshot: List[Dict], now: int) -> Dict[str, int]:
        seen = set()
        position_rows = []
        opened = closed = 0

        for flight in snapshot:
            icao = (flight.get("icao24") or "").lower()
            if not icao:
                continue
            seen.add(icao)
            lat = flight.get("latitude")
            lon = flight.get("longitude")
            cur_og = bool(flight.get("on_ground"))
            prev_og = self.last_on_ground.get(icao)

            position_rows.append({
                "icao24": icao, "ts": now, "lat": lat, "lon": lon,
                "alt_m": flight.get("altitude"), "speed": flight.get("speed"),
                "heading": flight.get("heading"), "on_ground": int(cur_og),
                "callsign": flight.get("callsign"),
                "classification": flight.get("classification"),
                "score": flight.get("classification_score"),})

            if icao in self.active:
                event = self.active[icao]
                self._apply_point(event, flight, icao, now)
                if prev_og is False and cur_og is True:
                    self._close(event, icao, now, "air_to_ground")
                    closed += 1
                else:
                    if prev_og is True and cur_og is False:
                        event["start_reason"] = "ground_to_air"
                        event["first_seen_ts"] = now
                    db.upsert_active_event(self.conn, event)
            else:
                self._open(flight, icao, now, cur_og)
                opened += 1

            self.last_on_ground[icao] = cur_og
            if lat is not None and lon is not None:
                self.last_pos[icao] = (lat, lon)

        db.insert_positions(self.conn, position_rows)

        timeout = Config.LANDING_TIMEOUT_MIN * 60
        for icao, event in list(self.active.items()):
            if icao not in seen and (now - event["last_seen_ts"]) > timeout:
                self._close(event, icao, event["last_seen_ts"], "signal_lost")
                closed += 1

        return {"tracked": len(seen), "opened": opened,
                "closed": closed, "active": len(self.active)}

    def close_all(self, now: int) -> None:
        for event in self.active.values():
            db.upsert_active_event(self.conn, event)


    def _open(self, flight: Dict, icao: str, now: int, on_ground: bool) -> None:
        lat = flight.get("latitude")
        lon = flight.get("longitude")
        alt = flight.get("altitude")
        base = flight.get("base_name") or "None"
        typecode = (flight.get("aircraft_type") or "").upper()
        info = Config.TRACKED_AIRCRAFT_TYPES.get(typecode)

        event = {
            "event_id": None,
            "icao24": icao,
            "callsign": flight.get("callsign") or "",
            "aircraft_type": typecode,
            "aircraft_name": info["name"] if info else "",
            "classification": flight.get("classification"),
            "score": flight.get("classification_score", 0),
            "first_seen_ts": now,
            "last_seen_ts": now,
            "duration_min": 0.0,
            "start_reason": "on_ground" if on_ground else "entered_region",
            "end_reason": None,
            "origin_lat": lat, "origin_lon": lon,
            "origin_base": base, "origin_airport": flight.get("origin_airport") or "",
            "dest_lat": lat, "dest_lon": lon,
            "dest_base": base, "dest_airport": flight.get("dest_airport") or "",
            "n_points": 1,
            "max_alt_m": alt, "min_alt_m": alt,
            "max_speed": flight.get("speed") or 0,
            "path_distance_km": 0.0,
            "bases_visited": base if base != "None" else "",
            "bz_price_start": db.price_asof(self.conn, "BZ=F", now),
            "cl_price_start": db.price_asof(self.conn, "CL=F", now),
            "bz_price_end": None, "cl_price_end": None,
            "status": "active",
            "updated_ts": now,}
        
        event["event_id"] = db.upsert_active_event(self.conn, event)
        self.active[icao] = event

    def _apply_point(self, event: Dict, flight: Dict, icao: str, now: int) -> None:
        lat = flight.get("latitude")
        lon = flight.get("longitude")
        alt = flight.get("altitude")
        speed = flight.get("speed") or 0
        base = flight.get("base_name") or "None"

        prev = self.last_pos.get(icao)
        if prev and lat is not None and lon is not None:
            event["path_distance_km"] = (event.get("path_distance_km") or 0.0) + \
                geodesic(prev, (lat, lon)).km

        event["last_seen_ts"] = now
        event["dest_lat"], event["dest_lon"] = lat, lon
        event["dest_base"] = base
        if flight.get("dest_airport"):
            event["dest_airport"] = flight["dest_airport"]
        event["n_points"] = (event.get("n_points") or 0) + 1

        if alt is not None:
            event["max_alt_m"] = max(event.get("max_alt_m") if event.get("max_alt_m") is not None else alt, alt)
            event["min_alt_m"] = min(event.get("min_alt_m") if event.get("min_alt_m") is not None else alt, alt)
        event["max_speed"] = max(event.get("max_speed") or 0, speed)

        if base and base != "None":
            visited = set(filter(None, (event.get("bases_visited") or "").split("|")))
            visited.add(base)
            event["bases_visited"] = "|".join(sorted(visited))

        event["duration_min"] = round((now - event["first_seen_ts"]) / 60.0, 1)
        event["updated_ts"] = now

    def _close(self, event: Dict, icao: str, end_ts: int, reason: str) -> None:
        event["end_reason"] = reason
        event["last_seen_ts"] = end_ts
        event["duration_min"] = round((end_ts - event["first_seen_ts"]) / 60.0, 1)
        event["bz_price_end"] = db.price_asof(self.conn, "BZ=F", end_ts)
        event["cl_price_end"] = db.price_asof(self.conn, "CL=F", end_ts)
        db.close_event(self.conn, event)
        self.active.pop(icao, None)
        self.last_pos.pop(icao, None)
        self.last_on_ground.pop(icao, None)
        self.logger.info(
            f"Closed {icao} ({event.get('aircraft_name') or event.get('aircraft_type') or '?'}) "
            f"reason={reason} dur={event['duration_min']}min dist={event['path_distance_km']:.0f}km")
