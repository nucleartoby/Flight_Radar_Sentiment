import requests
import time
import math
from datetime import datetime
from typing import List, Dict, Optional

from config.settings import Config


class OpenSkyAPICollector:

    def __init__(self):
        self.base_url = Config.OPENSKY_BASE_URL
        self.delay = Config.REQUEST_DELAY
        self.max_retries = Config.MAX_RETRIES

        self._token: Optional[str] = None
        self._token_expiry: float  = 0.0

        self.session = requests.Session()
        self._apply_auth()


    def _has_oauth2(self) -> bool:
        return bool(Config.OPENSKY_CLIENT_ID and Config.OPENSKY_CLIENT_SECRET)

    def _has_basic_auth(self) -> bool:
        return bool(Config.OPENSKY_USERNAME and Config.OPENSKY_PASSWORD)

    def _fetch_token(self) -> Optional[str]:
            resp = requests.post(Config.OPENSKY_TOKEN_URL, data={"grant_type":"client_credentials","client_id":Config.OPENSKY_CLIENT_ID,"client_secret": Config.OPENSKY_CLIENT_SECRET,},timeout=15,)
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._token_expiry = time.time() + payload.get("expires_in", 300) - 30

            return self._token

    def _apply_auth(self):
        if self._has_oauth2():
            token = self._fetch_token()
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif self._has_basic_auth():
            self.session.auth = (Config.OPENSKY_USERNAME, Config.OPENSKY_PASSWORD)

    def _ensure_token_fresh(self):
        if self._has_oauth2() and time.time() >= self._token_expiry:
            token = self._fetch_token()
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})


    def get_flights_in_region(self, bounds: Dict[str, float]) -> List[Dict]:
        self._ensure_token_fresh()
        flights = []

        for retry in range(self.max_retries):
            try:
                params = {'lamin': bounds['south'],'lamax': bounds['north'],'lomin': bounds['west'],'lomax': bounds['east'],}
                response = self.session.get(f"{self.base_url}/states/all", params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data or 'states' not in data or not data['states']:
                    break

                for state in data['states']:
                    if len(state) >= 17:
                        flight_info = self._parse_state_vector(state)
                        if flight_info:
                            flights.append(flight_info)
                break

            except requests.exceptions.RequestException:
                if retry < self.max_retries - 1:
                    time.sleep(self.delay * (retry + 1))
            except Exception:
                break

        time.sleep(self.delay)
        return flights

    def _parse_state_vector(self, state: List) -> Optional[Dict]:
        if state[5] is None or state[6] is None:
            return None

        flight_info = {
            'flight_id':      state[0],
            'icao24':         state[0],
            'callsign':       state[1].strip() if state[1] else '',
            'origin_country': state[2],
            'time_position':  state[3],
            'last_contact':   state[4],
            'longitude':      float(state[5]),
            'latitude':       float(state[6]),
            'baro_altitude':  state[7],
            'on_ground':      state[8],
            'velocity':       state[9],
            'true_track':     state[10],
            'vertical_rate':  state[11],
            'sensors':        state[12],
            'geo_altitude':   state[13],
            'squawk':         state[14],
            'spi':            state[15],
            'position_source':state[16],
            'timestamp':      datetime.now().timestamp(),
            'api_source':     'opensky',
        }
        flight_info['altitude'] = flight_info['baro_altitude']
        flight_info['speed']    = flight_info['velocity']
        flight_info['heading']  = flight_info['true_track']
        return flight_info

    def get_middle_east_flights(self) -> List[Dict]:
        return self.get_flights_in_region({'north': 38.0, 'south': 12.0, 'west': 32.0, 'east': 60.0})


    def get_flights_around_base(self, latitude: float, longitude: float,
                                radius_km: int = 100) -> List[Dict]:
        return self.get_flights_in_region(self._calculate_bounding_box(latitude, longitude, radius_km))


    def _calculate_bounding_box(self, lat: float, lon: float,
                                radius_km: int) -> Dict[str, float]:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        return { 'north': lat + lat_delta,'south': lat - lat_delta,'east':  lon + lon_delta,'west':  lon - lon_delta,}


    def filter_military_flights(self, flights: List[Dict]) -> List[Dict]:
        return [f for f in flights if self._is_military_flight(f)]

    def _is_military_flight(self, flight: Dict) -> bool:
        callsign = flight.get('callsign', '').upper()
        icao24 = flight.get('icao24', '').upper()
        squawk = flight.get('squawk', '') or ''

        if squawk == '7777':
            flight['military_reason'] = "Squawk 7777 (military intercept)"
            return True

        for token in Config.MILITARY_CALLSIGNS:
            if token in callsign:
                flight['military_reason'] = f"Callsign contains {token}"
                return True

        icao_prefix = icao24[:2] if len(icao24) >= 2 else ''
        if icao_prefix in Config.MILITARY_ICAO_CODES:
            flight['military_reason'] = f"Military ICAO prefix: {icao_prefix}"
            return True

        digits_only = callsign.isdigit()
        if digits_only and len(callsign) <= 4:
            flight['military_reason'] = "Short numeric callsign (possible state/military)"
            return True

        return False


    def get_api_status(self) -> Dict:
        self._ensure_token_fresh()
        try:
            response = self.session.get(
                f"{self.base_url}/states/all",
                params={'lamin': 0, 'lamax': 1, 'lomin': 0, 'lomax': 1},
                timeout=10,
            )
            return {
                'api_accessible': response.status_code == 200,
                'authenticated':  self._has_oauth2() or self._has_basic_auth(),
                'auth_method':    'oauth2' if self._has_oauth2() else
                                  ('basic' if self._has_basic_auth() else 'anonymous'),
                'daily_limit':    4000 if (self._has_oauth2() or self._has_basic_auth()) else 100,
                'base_url':       self.base_url,
                'response_code':  response.status_code,
                'timestamp':      datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                'api_accessible': False,
                'authenticated':  self._has_oauth2() or self._has_basic_auth(),
                'error':          str(e),
                'timestamp':      datetime.now().isoformat(),
            }

    def close(self):
        self.session.close()
