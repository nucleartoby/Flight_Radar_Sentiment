import time
import logging
from typing import Dict, List

from config.settings import Config

logger = logging.getLogger("snapshot")


def _enrich(flight: Dict, collector, monitor) -> Dict:
    nearby = monitor.find_nearest_base(
        flight.get("latitude", 0), flight.get("longitude", 0))
    
    flight["near_base"] = nearby is not None
    flight["base_name"] = nearby["name"] if nearby else "None"
    flight["base_type"] = nearby.get("type", "") if nearby else ""
    flight["base_priority"] = nearby.get("priority", 0) if nearby else 0
    flight.update(collector.classify_flight(flight))
    return flight


def _raw_poll(collector) -> List[Dict]:
    for attempt in range(Config.SNAPSHOT_RETRIES):
        flights = collector.get_middle_east_flights()
        if flights:
            return flights
        if attempt < Config.SNAPSHOT_RETRIES - 1:
            time.sleep(Config.SNAPSHOT_RETRY_WAIT)
    return []


def collect_snapshot(collector, monitor) -> List[Dict]:
    by_icao: Dict[str, Dict] = {}

    for sample in range(max(1, Config.SNAPSHOT_SAMPLES)):
        for flight in _raw_poll(collector):
            icao = (flight.get("icao24") or "").lower()
            if not icao:
                continue
            by_icao[icao] = flight  # later sample wins freshest position
        if sample < Config.SNAPSHOT_SAMPLES - 1:
            time.sleep(Config.SNAPSHOT_RETRY_WAIT)

    enriched = [_enrich(f, collector, monitor) for f in by_icao.values()]

    if Config.TRACK_ONLY_MILITARY:
        enriched = [f for f in enriched if f.get("is_military")]

    logger.info(f"Snapshot: {len(by_icao)} unique aircraft, {len(enriched)} tracked.")
    return enriched
