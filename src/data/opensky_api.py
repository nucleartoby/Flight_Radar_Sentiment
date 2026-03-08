import requests
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
import math
from datetime import datetime
from config.settings import Config


class OpenSkyAPICollector:
    
    def __init__(self):
        self.base_url = Config.OPENSKY_BASE_URL
        self.delay = Config.REQUEST_DELAY
        self.max_retries = Config.MAX_RETRIES

        self.session = requests.Session()
        if Config.OPENSKY_USERNAME and Config.OPENSKY_PASSWORD:
            self.session.auth = (Config.OPENSKY_USERNAME, Config.OPENSKY_PASSWORD)
    
    def get_flights_in_region(self, bounds: Dict[str, float]) -> List[Dict]:
        flights = []
        
        for retry in range(self.max_retries):
            try:
                url = f"{self.base_url}/states/all"
                params = {
                    'lamin': bounds['south'],
                    'lamax': bounds['north'], 
                    'lomin': bounds['west'],
                    'lomax': bounds['east']
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or 'states' not in data or not data['states']:
                    return flights

                for state in data['states']:
                    if len(state) >= 17:
                        flight_info = self._parse_state_vector(state)
                        if flight_info:
                            flights.append(flight_info)
                
                break
                
            except requests.exceptions.RequestException:
                if retry < self.max_retries - 1:
                    time.sleep(self.delay * (retry + 1))
                else:
                    break
            except Exception:
                break
        
        time.sleep(self.delay)
        return flights
    
    def _parse_state_vector(self, state: List) -> Optional[Dict]:
            if state[5] is None or state[6] is None:
                return None
            
            flight_info = {
                'flight_id': state[0],  # ICAO24 as flight ID
                'icao24': state[0],
                'callsign': state[1].strip() if state[1] else '',
                'origin_country': state[2],
                'time_position': state[3],
                'last_contact': state[4],
                'longitude': float(state[5]),
                'latitude': float(state[6]),
                'baro_altitude': state[7],
                'on_ground': state[8],
                'velocity': state[9],
                'true_track': state[10],  # heading
                'vertical_rate': state[11],
                'sensors': state[12],
                'geo_altitude': state[13],
                'squawk': state[14],
                'spi': state[15],
                'position_source': state[16],
                'timestamp': datetime.now().timestamp(),
                'api_source': 'opensky'
            }
            
            flight_info['altitude'] = flight_info['baro_altitude']
            flight_info['speed'] = flight_info['velocity']
            flight_info['heading'] = flight_info['true_track']
            
            return flight_info

    
    def get_middle_east_flights(self) -> List[Dict]:
        bounds = {
            'north': 37.0,
            'south': 24.0,
            'west': 38.0,
            'east': 55.0
        }
        return self.get_flights_in_region(bounds)
    
    def get_flights_around_base(self, latitude: float, longitude: float, radius_km: int = 100) -> List[Dict]:
        bounds = self._calculate_bounding_box(latitude, longitude, radius_km)
        return self.get_flights_in_region(bounds)
    
    def _calculate_bounding_box(self, lat: float, lon: float, radius_km: int) -> Dict[str, float]:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        
        return {
            'north': lat + lat_delta,
            'south': lat - lat_delta,
            'east': lon + lon_delta,
            'west': lon - lon_delta
        }
    
    def filter_military_flights(self, flights: List[Dict]) -> List[Dict]:
        military_flights = []
        
        for flight in flights:
            if self._is_military_flight(flight):
                military_flights.append(flight)
        
        return military_flights
    
    def _is_military_flight(self, flight: Dict) -> bool:
        callsign = flight.get('callsign', '').upper()
        icao24 = flight.get('icao24', '').upper()
        origin_country = flight.get('origin_country', '')

        for military_callsign in Config.MILITARY_CALLSIGNS:
            if military_callsign in callsign:
                flight['military_reason'] = f"Callsign contains {military_callsign}"
                return True

        if origin_country == 'United States':
            icao_prefix = icao24[:2] if len(icao24) >= 2 else ''
            if icao_prefix in Config.MILITARY_ICAO_CODES:
                flight['military_reason'] = f"Military ICAO prefix: {icao_prefix}"
                return True

        military_patterns = [
            'USAF', 'ARMY', 'NAVY', 'USMC', 'USCG',
            'AFO', 'SAM', 'GLEX', 'REACH', 'CONVOY'
        ]
        
        for pattern in military_patterns:
            if pattern in callsign:
                flight['military_reason'] = f"Military pattern: {pattern}"
                return True
        
        return False
    
    def get_api_status(self) -> Dict:
        try:
            url = f"{self.base_url}/states/all"
            params = {'lamin': 0, 'lamax': 1, 'lomin': 0, 'lomax': 1}
            
            response = self.session.get(url, params=params, timeout=10)
            
            status = {
                'api_accessible': response.status_code == 200,
                'authenticated': bool(Config.OPENSKY_USERNAME and Config.OPENSKY_PASSWORD),
                'daily_limit': 4000 if Config.OPENSKY_USERNAME else 100,
                'base_url': self.base_url,
                'response_code': response.status_code,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            return {
                'api_accessible': False,
                'authenticated': bool(Config.OPENSKY_USERNAME and Config.OPENSKY_PASSWORD),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def close(self):
        if hasattr(self.session, 'close'):
            self.session.close()
