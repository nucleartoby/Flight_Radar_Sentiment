import json
import logging
from typing import List, Dict, Tuple, Optional

from geopy.distance import geodesic
from config.settings import Config


class BaseMonitor:

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.bases  = self._load_bases()
        # Global fallback radius used only when a base has no radius_km field.
        self._default_radius_km  = Config.BASE_RADIUS_KM
        # Flat callsign set kept for the legacy is_military_aircraft() method.
        self._military_callsigns = set(Config.MILITARY_CALLSIGNS)

    def _load_bases(self) -> List[Dict]:
        try:
            with open('config/military_bases.json', 'r') as f:
                return json.load(f)['military_bases']
        except Exception as exc:
            self.logger.error(f"Failed to load military_bases.json: {exc}")
            return []


    def find_nearest_base(self, aircraft_lat: float,
                          aircraft_lon: float) -> Optional[Dict]:

        aircraft_pos = (aircraft_lat, aircraft_lon)
        for base in self.bases:
            radius = base.get('radius_km', self._default_radius_km)
            base_pos = (base['latitude'], base['longitude'])
            if geodesic(aircraft_pos, base_pos).kilometers <= radius:
                return base
        return None


    def is_near_base(self, aircraft_lat: float,
                     aircraft_lon: float) -> Tuple[bool, str]:
        base = self.find_nearest_base(aircraft_lat, aircraft_lon)
        if base:
            return True, base['name']
        return False, ""


    def is_military_aircraft(self, callsign: str) -> bool:
        if not callsign:
            return False
        cs = callsign.upper()
        return any(token in cs for token in self._military_callsigns)


    def categorise_activity(self, flight_data: List[Dict]) -> Dict[str, int]:
        activity = {base['name']: 0 for base in self.bases}
        total_military = 0
        for flight in flight_data:
            if self.is_military_aircraft(flight.get('callsign', '')):
                base = self.find_nearest_base(
                    flight.get('latitude', 0), flight.get('longitude', 0))
                if base:
                    activity[base['name']] += 1
                    total_military += 1
        activity['total_military_flights'] = total_military
        return activity