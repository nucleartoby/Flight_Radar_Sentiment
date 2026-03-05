import json
import pandas as pd
from typing import List, Dict, Tuple
from geopy.distance import geodesic
import logging
from config.settings import Config

class BaseMonitor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)  # <-- Set logger first
        self.bases = self._load_bases()
        self.radius_km = Config.BASE_RADIUS_KM
        self.military_callsigns = Config.MILITARY_CALLSIGNS

    def load_bases(self) -> List[Dict]:
        try:
            with open('config/military_bases.json', 'r') as f:
                data = json.load(f)
                return data['military_bases']
        except Exception as e:
            self.logger.error(f"Error loading bases; {e}")
            return []
        
    def is_near_base(self, aircraft_lat: float, aircraft_lon: float) -> Tuple[bool, str]:
        aircraft_pos = (aircraft_lat, aircraft_lon)

        for base in self.bases:

            base_pos = (base['latitude'], base['longitude'])
            distance = geodesic(aircraft_pos, base_pos).kilometers
            
            if distance <= self.radius_km:
                return True, base['name']
        return False, ""
    
    def is_military_aircraft(self, callsign: str) -> bool:
        if not callsign:
            return False
        
        callsign_upper = callsign.upper()
        return any(mil_sign in callsign_upper for mil_sign in self.military_callsigns)
    
    def categorise_activity(self, flight_data: List[Dict]) -> Dict[str, int]:
        activity = {base['name']: 0 for base in self.bases}
        total_military = 0
        
        for flight in flight_data:
            if self.is_military_aircraft(flight.get('callsign', '')):
                is_near, base_name = self.is_near_base(
                    flight.get('latitude', 0), 
                    flight.get('longitude', 0)
                )
                if is_near:
                    activity[base_name] += 1
                    total_military += 1
        
        activity['total_military_flights'] = total_military
        return activity