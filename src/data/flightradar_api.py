import re
from datetime import datetime
from typing import List, Dict, Optional

from FlightRadarAPI import FlightRadar24API
from config.settings import Config


_IATA_RE = re.compile(r'^[A-Z]{2,3}\d{2,4}$')


class FlightRadarCollector:

    def __init__(self):
        self._api = FlightRadar24API()
        self._logged_in = False
        self._try_login()

    def _try_login(self):
        email = Config.FR24_EMAIL
        password = Config.FR24_PASSWORD

    def get_flights_in_region(self, bounds: Dict[str, float]) -> List[Dict]:
        bounds_str = (f"{bounds['north']},{bounds['south']}"
            f",{bounds['west']},{bounds['east']}")

        raw = self._api.get_flights(bounds=bounds_str)
        results = [p for f in raw if (p := self._parse_flight(f)) is not None]
        return results

    def _parse_flight(self, f) -> Optional[Dict]:
        lat = getattr(f, 'latitude', None)
        lon = getattr(f, 'longitude', None)
        if lat is None or lon is None:
            return None

        alt_ft = getattr(f, 'altitude', 0) or 0
        spd_kt = getattr(f, 'ground_speed', 0) or 0
        vs_fpm = getattr(f, 'vertical_speed', 0) or 0

        flight = {
            'icao24': (getattr(f, 'icao_24bit', '') or '').lower(),
            'callsign': (getattr(f, 'callsign', '') or '').strip(),
            'registration': (getattr(f, 'registration', '') or ''),
            'aircraft_type': (getattr(f, 'aircraft_code', '') or '').upper(),
            'airline_iata': (getattr(f, 'airline_iata', '') or '').upper(),
            'origin_airport': (getattr(f, 'origin_airport_iata', '') or ''),
            'dest_airport': (getattr(f, 'destination_airport_iata', '') or ''),
            'latitude': float(lat),
            'longitude': float(lon),
            'baro_altitude': round(alt_ft * 0.3048, 1),
            'altitude_ft': alt_ft,
            'on_ground': bool(getattr(f, 'on_ground', False)),
            'velocity': round(spd_kt * 0.514444, 1),
            'speed_kt': spd_kt,
            'true_track': getattr(f, 'heading', 0) or 0,
            'vertical_rate': round(vs_fpm * 0.00508, 2),
            'squawk': str(getattr(f, 'squawk', '') or ''),
            'timestamp': datetime.now().timestamp(),
            'api_source': 'flightradar24',
            'altitude': round(alt_ft * 0.3048, 1),
            'speed': round(spd_kt * 0.514444, 1),
            'heading': getattr(f, 'heading', 0) or 0,}

        return flight

    def get_middle_east_flights(self) -> List[Dict]:
        return self.get_flights_in_region(
            {'north': 38.0, 'south': 12.0, 'west': 32.0, 'east': 60.0})

    def classify_flight(self, flight: Dict) -> Dict:
        callsign = (flight.get('callsign') or '').strip().upper()
        icao24 = (flight.get('icao24') or '').upper()
        squawk = str(flight.get('squawk') or '')
        typecode = (flight.get('aircraft_type') or '').upper()
        airline_iata = (flight.get('airline_iata') or '').upper()
        altitude = flight.get('baro_altitude') or 0
        on_ground = bool(flight.get('on_ground', False))
        base_type = flight.get('base_type', '')
        base_pri = int(flight.get('base_priority', 0))

        score = 0
        evidence = []
        has_gov_logistics = False
        has_strong = False

        type_info = Config.TRACKED_AIRCRAFT_TYPES.get(typecode)
        if type_info:
            boost = type_info['weight_boost']
            score += boost
            evidence.append(f'tracked_type:{typecode}({type_info["name"]},w={boost})')
            has_strong = True

        # Squawk 7777
        if squawk == '7777':
            score += 50
            evidence.append('squawk_7777')
            has_strong = True

        # Callsign tiers
        matched = False
        for token in Config.STRONG_MIL_CALLSIGNS:
            if token in callsign:
                w = Config.CALLSIGN_WEIGHTS['strong_mil']
                score += w
                evidence.append(f'strong_mil_callsign:{token}(w={w})')
                has_strong = True
                matched = True
                break

        if not matched:
            for token in Config.GOV_LOGISTICS_CALLSIGNS:
                if token in callsign:
                    w = Config.CALLSIGN_WEIGHTS['gov_logistics']
                    score += w
                    evidence.append(f'gov_logistics_callsign:{token}(w={w})')
                    has_gov_logistics = True
                    has_strong = True
                    matched = True
                    break

        if not matched:
            for token in Config.WEAK_GOV_CALLSIGNS:
                if token in callsign:
                    w = Config.CALLSIGN_WEIGHTS['weak_gov']
                    score += w
                    evidence.append(f'weak_gov_callsign:{token}(w={w})')
                    break

        icao_prefix = icao24[:2] if len(icao24) >= 2 else ''
        icao_w = Config.ICAO_BLOCK_WEIGHTS.get(icao_prefix, 0)
        if icao_w:
            score += icao_w
            evidence.append(f'icao_block:{icao_prefix}(weak,w={icao_w})')

        # Base proximity contextual only
        if flight.get('near_base') and base_type:
            type_w = Config.BASE_TYPE_WEIGHTS.get(base_type, 8)
            prox_w = max(1, int(type_w * base_pri / 5))
            score += prox_w
            evidence.append(
                f'near_base:{flight.get("base_name")}(type={base_type},pri={base_pri},w={prox_w})')

        if not callsign and not on_ground and altitude and float(altitude) > 3000:
            score += 14
            evidence.append('no_callsign_airborne')
        elif not callsign:
            score += 5
            evidence.append('no_callsign')

        if callsign.isdigit() and 1 <= len(callsign) <= 4:
            score += 10
            evidence.append('short_numeric_callsign')

        # FR24 provides airline_iata directly stronger than prefix guessing
        if airline_iata and airline_iata in Config.COMMERCIAL_AIRLINE_IATA:
            neg = Config.NEGATIVE_WEIGHTS['commercial_prefix']
            score += neg
            evidence.append(f'commercial_airline:{airline_iata}(w={neg})')
        else:
            for prefix in Config.COMMERCIAL_CALLSIGN_PREFIXES:
                if callsign.startswith(prefix):
                    neg = Config.NEGATIVE_WEIGHTS['commercial_prefix']
                    score += neg
                    evidence.append(f'commercial_prefix:{prefix}(w={neg})')
                    break

        if callsign and _IATA_RE.match(callsign):
            if not any(t in callsign for t in Config.STRONG_MIL_CALLSIGNS):
                neg = Config.NEGATIVE_WEIGHTS['iata_format']
                score += neg
                evidence.append(f'iata_format(w={neg})')

        # Classify
        score = max(0, min(100, score))
        thr = Config.CONFIDENCE_THRESHOLDS

        if score >= thr['likely_military']:
            classification = 'likely_military'
        elif score >= thr['gov_logistics'] and has_gov_logistics:
            classification = 'gov_logistics'
        elif score >= thr['unknown']:
            classification = 'unknown'
        else:
            classification = 'likely_civilian'

        if score >= 75:
            confidence_band = 'high'
        elif score >= 50:
            confidence_band = 'medium'
        elif score >= 25:
            confidence_band = 'low'
        else:
            confidence_band = 'civilian'

        has_proximity = any('near_base' in e for e in evidence)
        if score >= thr['likely_military'] and has_strong:
            verification_status = 'high_confidence_rule_match'
        elif has_proximity and len(evidence) > 1:
            verification_status = 'cross_source_candidate'
        else:
            verification_status = 'unverified'

        evidence_str = ' | '.join(evidence) if evidence else 'none'
        return {
            'classification': classification,
            'classification_score': score,
            'confidence_band': confidence_band,
            'evidence_flags': evidence_str,
            'verification_status': verification_status,
            'military_reason': evidence_str,
            'is_military': classification in ('likely_military', 'gov_logistics', 'unknown'),}

    # Status

    def get_api_status(self) -> Dict:

        self._api.get_flights(bounds="25,26,51,52")
        return {
            'api_accessible': True,
            'auth_method': 'fr24_authenticated' if self._logged_in else 'fr24_anonymous',
            'timestamp': datetime.now().isoformat(),}

    def close(self):
        if self._logged_in:
            self._api.logout()