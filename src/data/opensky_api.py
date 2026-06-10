import re
import json
import os
import requests
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Optional

from config.settings import Config

logger = logging.getLogger(__name__)

# Regex: typical IATA commercial format  e.g. UAE232, QTR541, BAW019
_IATA_RE = re.compile(r'^[A-Z]{2,3}\d{2,4}$')


class OpenSkyAPICollector:

    def __init__(self):
        self.base_url    = Config.OPENSKY_BASE_URL
        self.delay       = Config.REQUEST_DELAY
        self.max_retries = Config.MAX_RETRIES

        self._token: Optional[str] = None
        self._token_expiry: float  = 0.0

        self._metadata_cache: Dict[str, Optional[Dict]] = {}
        self._metadata_cache_path = 'data/processed/metadata_cache.json'
        self._load_metadata_cache()

        self.session = requests.Session()
        self._apply_auth()

    # ── Authentication ────────────────────────────────────────────────────────

    def _has_oauth2(self) -> bool:
        return bool(Config.OPENSKY_CLIENT_ID and Config.OPENSKY_CLIENT_SECRET)

    def _has_basic_auth(self) -> bool:
        return bool(Config.OPENSKY_USERNAME and Config.OPENSKY_PASSWORD)

    def _fetch_token(self) -> Optional[str]:
        try:
            resp = requests.post(
                Config.OPENSKY_TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     Config.OPENSKY_CLIENT_ID,
                    "client_secret": Config.OPENSKY_CLIENT_SECRET,
                },
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            expires_in = payload.get("expires_in", 300)
            self._token_expiry = time.time() + expires_in - 30
            logger.info(f"OpenSky token acquired (expires in {expires_in}s)")
            return self._token
        except Exception as exc:
            logger.error(f"Token fetch failed: {exc}")
            return None

    def _apply_auth(self):
        if self._has_oauth2():
            token = self._fetch_token()
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif self._has_basic_auth():
            self.session.auth = (Config.OPENSKY_USERNAME, Config.OPENSKY_PASSWORD)

    def _ensure_token_fresh(self):
        if self._has_oauth2() and time.time() >= self._token_expiry:
            logger.info("OpenSky token expired — refreshing…")
            token = self._fetch_token()
            if token:
                self.session.headers.update({"Authorization": f"Bearer {token}"})

    # ── Data fetching ─────────────────────────────────────────────────────────

    def get_flights_in_region(self, bounds: Dict[str, float]) -> List[Dict]:
        self._ensure_token_fresh()
        flights = []

        for retry in range(self.max_retries):
            try:
                params = {
                    'lamin':    bounds['south'],
                    'lamax':    bounds['north'],
                    'lomin':    bounds['west'],
                    'lomax':    bounds['east'],
                    'extended': 1,          # field 17 = category
                }
                response = self.session.get(
                    f"{self.base_url}/states/all", params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if not data or 'states' not in data or not data['states']:
                    break

                for state in data['states']:
                    flight_info = self._parse_state_vector(state)
                    if flight_info:
                        flights.append(flight_info)
                break

            except requests.exceptions.RequestException as exc:
                logger.warning(f"Request error (attempt {retry + 1}): {exc}")
                if retry < self.max_retries - 1:
                    time.sleep(self.delay * (retry + 1))
            except Exception as exc:
                logger.error(f"Unexpected error fetching flights: {exc}")
                break

        time.sleep(self.delay)
        return flights

    def _parse_state_vector(self, state: List) -> Optional[Dict]:
        """
        Parse an OpenSky state vector.
        Without extended=1: 17 fields (indices 0-16).
        With    extended=1: 18 fields (index 17 = category).
        """
        # Must have position
        if len(state) < 7 or state[5] is None or state[6] is None:
            return None

        def _safe(lst, idx, default=None):
            try:
                return lst[idx]
            except IndexError:
                return default

        flight = {
            'icao24':          _safe(state, 0, ''),
            'callsign':        (_safe(state, 1) or '').strip(),
            'origin_country':  _safe(state, 2, ''),
            'time_position':   _safe(state, 3),
            'last_contact':    _safe(state, 4),
            'longitude':       float(state[5]),
            'latitude':        float(state[6]),
            'baro_altitude':   _safe(state, 7),
            'on_ground':       _safe(state, 8, False),
            'velocity':        _safe(state, 9),
            'true_track':      _safe(state, 10),
            'vertical_rate':   _safe(state, 11),
            'sensors':         _safe(state, 12),
            'geo_altitude':    _safe(state, 13),
            'squawk':          _safe(state, 14, ''),
            'spi':             _safe(state, 15),
            'position_source': _safe(state, 16),
            'category':        _safe(state, 17),      # None when extended=1 absent
            'timestamp':       datetime.now().timestamp(),
            'api_source':      'opensky',
        }
        # Aliases expected by downstream feature engineering
        flight['altitude'] = flight['baro_altitude']
        flight['speed']    = flight['velocity']
        flight['heading']  = flight['true_track']
        return flight

    def get_middle_east_flights(self) -> List[Dict]:
        # Covers Israel / Levant (lon ~34) through Gulf of Oman (lon ~60)
        # and Turkey border (lat 38) down to Yemen / Red Sea (lat 12)
        return self.get_flights_in_region(
            {'north': 38.0, 'south': 12.0, 'west': 32.0, 'east': 60.0}
        )

    def get_flights_around_base(self, latitude: float, longitude: float,
                                radius_km: int = 100) -> List[Dict]:
        return self.get_flights_in_region(
            self._calculate_bounding_box(latitude, longitude, radius_km)
        )

    def _calculate_bounding_box(self, lat: float, lon: float,
                                radius_km: int) -> Dict[str, float]:
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        return {
            'north': lat + lat_delta,
            'south': lat - lat_delta,
            'east':  lon + lon_delta,
            'west':  lon - lon_delta,
        }

    # ── Confidence-based classifier ───────────────────────────────────────────

    def classify_flight(self, flight: Dict) -> Dict:
        """
        Score 0-100.  Evidence accumulates from independent signals; no single
        signal is sufficient for classification except very strong rule matches.

        Returns:
          classification      : 'likely_military' | 'gov_logistics' | 'unknown' | 'likely_civilian'
          classification_score: int 0-100
          confidence_band     : 'high' | 'medium' | 'low' | 'civilian'
          evidence_flags      : pipe-separated human-readable evidence trail
          verification_status : 'high_confidence_rule_match' | 'cross_source_candidate' | 'unverified'
          military_reason     : alias for evidence_flags (backward compat)
          is_military         : True for likely_military, gov_logistics, unknown
        """
        callsign  = (flight.get('callsign') or '').strip().upper()
        icao24    = (flight.get('icao24')   or '').upper()
        squawk    = str(flight.get('squawk') or '')
        category  = flight.get('category')
        altitude  = flight.get('baro_altitude') or 0
        on_ground = bool(flight.get('on_ground', False))
        base_type = flight.get('base_type', '')
        base_pri  = int(flight.get('base_priority', 0))

        score              = 0
        evidence           = []
        has_gov_logistics  = False
        has_strong_signal  = False   # any high-confidence positive hit

        # ── Squawk 7777 ───────────────────────────────────────────────────────
        if squawk == '7777':
            score += 50
            evidence.append('squawk_7777')
            has_strong_signal = True

        # ── Callsign tiers ────────────────────────────────────────────────────
        # Checked in descending priority; only the first matching tier fires.
        matched_callsign = False
        for token in Config.STRONG_MIL_CALLSIGNS:
            if token in callsign:
                w = Config.CALLSIGN_WEIGHTS['strong_mil']
                score += w
                evidence.append(f'strong_mil_callsign:{token}(w={w})')
                has_strong_signal = True
                matched_callsign  = True
                break

        if not matched_callsign:
            for token in Config.GOV_LOGISTICS_CALLSIGNS:
                if token in callsign:
                    w = Config.CALLSIGN_WEIGHTS['gov_logistics']
                    score += w
                    evidence.append(f'gov_logistics_callsign:{token}(w={w})')
                    has_gov_logistics = True
                    has_strong_signal = True
                    matched_callsign  = True
                    break

        if not matched_callsign:
            for token in Config.WEAK_GOV_CALLSIGNS:
                if token in callsign:
                    w = Config.CALLSIGN_WEIGHTS['weak_gov']
                    score += w
                    evidence.append(f'weak_gov_callsign:{token}(w={w})')
                    break

        # ── ICAO24 hex block ──────────────────────────────────────────────────
        # Country block allocation is NOT proof of military status.
        # AE/AF cover the entire US military ICAO range including many
        # non-combat, training, and contractor-operated aircraft.
        # Weight intentionally low; never sufficient alone.
        icao_prefix = icao24[:2] if len(icao24) >= 2 else ''
        icao_w = Config.ICAO_BLOCK_WEIGHTS.get(icao_prefix, 0)
        if icao_w:
            score += icao_w
            evidence.append(f'icao_block:{icao_prefix}(weak,w={icao_w})')

        # ── OpenSky category ─────────────────────────────────────────────────
        # Useful corroborating signal, but not conclusive:
        #   cat 7 (high-perf) can include fast bizjets; cat 0 is the default
        #   for many military transponders but also for aircraft with no ADS-B.
        cat_w = Config.CATEGORY_WEIGHTS.get(category, 0) if category is not None else 0
        if cat_w:
            score += cat_w
            evidence.append(f'category:{category}(w={cat_w})')

        # ── Base proximity ────────────────────────────────────────────────────
        # Contextual evidence only.  Commercial airlines overfly or approach
        # high-traffic bases (Al Udeid, Bahrain, Al Dhafra) routinely.
        # Weight is scaled by base type and base priority (1-5).
        if flight.get('near_base') and base_type:
            type_w = Config.BASE_TYPE_WEIGHTS.get(base_type, 8)
            prox_w = max(1, int(type_w * base_pri / 5))
            score  += prox_w
            evidence.append(
                f'near_base:{flight.get("base_name")}(type={base_type},pri={base_pri},w={prox_w})'
            )

        # ── No-callsign rules ─────────────────────────────────────────────────
        if not callsign and not on_ground and altitude and float(altitude) > 3000:
            score += 14
            evidence.append('no_callsign_airborne')
        elif not callsign:
            score += 5
            evidence.append('no_callsign')

        if callsign.isdigit() and 1 <= len(callsign) <= 4:
            score += 10
            evidence.append('short_numeric_callsign')

        # ── Negative signals ──────────────────────────────────────────────────
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

        # ── Final score ───────────────────────────────────────────────────────
        score = max(0, min(100, score))
        thr   = Config.CONFIDENCE_THRESHOLDS

        if score >= thr['likely_military']:
            classification = 'likely_military'
        elif score >= thr['gov_logistics'] and has_gov_logistics:
            # gov_logistics requires a positive gov-logistics callsign hit AND
            # score above the threshold — score alone is not enough.
            classification = 'gov_logistics'
        elif score >= thr['unknown']:
            classification = 'unknown'
        else:
            classification = 'likely_civilian'

        # ── Confidence band ───────────────────────────────────────────────────
        if score >= 75:
            confidence_band = 'high'
        elif score >= 50:
            confidence_band = 'medium'
        elif score >= 25:
            confidence_band = 'low'
        else:
            confidence_band = 'civilian'

        # ── Verification status ───────────────────────────────────────────────
        has_category_signal  = cat_w > 0
        has_proximity_signal = any('near_base' in e for e in evidence)

        if score >= thr['likely_military'] and has_strong_signal:
            verification_status = 'high_confidence_rule_match'
        elif has_category_signal and has_proximity_signal:
            verification_status = 'cross_source_candidate'
        else:
            verification_status = 'unverified'

        evidence_str = ' | '.join(evidence) if evidence else 'none'

        return {
            'classification':       classification,
            'classification_score': score,
            'confidence_band':      confidence_band,
            'evidence_flags':       evidence_str,
            'verification_status':  verification_status,
            'military_reason':      evidence_str,     # backward compat alias
            'is_military':          classification in ('likely_military', 'gov_logistics', 'unknown'),
        }

    # ── Backward-compat thin wrapper ──────────────────────────────────────────

    def _is_military_flight(self, flight: Dict) -> bool:
        result = self.classify_flight(flight)
        flight.update(result)
        return result['is_military']

    def filter_military_flights(self, flights: List[Dict]) -> List[Dict]:
        return [f for f in flights if self._is_military_flight(f)]

    # ── Status ────────────────────────────────────────────────────────────────

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
        except Exception as exc:
            return {
                'api_accessible': False,
                'authenticated':  self._has_oauth2() or self._has_basic_auth(),
                'error':          str(exc),
                'timestamp':      datetime.now().isoformat(),
            }

    # ── Aircraft type metadata ────────────────────────────────────────────────

    def _load_metadata_cache(self):
        """Load persisted ICAO24 → metadata cache from disk (survives restarts)."""
        try:
            if os.path.exists(self._metadata_cache_path):
                with open(self._metadata_cache_path) as f:
                    self._metadata_cache = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load metadata cache: {exc}")

    def _save_metadata_cache(self):
        """Persist the in-memory cache so subsequent runs skip known ICAO24s."""
        try:
            os.makedirs(os.path.dirname(self._metadata_cache_path), exist_ok=True)
            with open(self._metadata_cache_path, 'w') as f:
                json.dump(self._metadata_cache, f)
        except Exception as exc:
            logger.warning(f"Could not save metadata cache: {exc}")

    def fetch_aircraft_metadata(self, icao24: str) -> Optional[Dict]:
        """
        Fetch aircraft type, registration and operator from the OpenSky metadata
        endpoint.  Results are cached in memory and on disk so repeated calls for
        the same ICAO24 cost nothing after the first lookup.

        Returns None if the aircraft is unknown or the request fails.
        """
        key = icao24.lower().strip()
        if key in self._metadata_cache:
            return self._metadata_cache[key]

        self._ensure_token_fresh()
        try:
            resp = self.session.get(
                f"{self.base_url}/metadata/aircraft/icao/{key}",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._metadata_cache[key] = data
                return data
            # 404 = unknown ICAO24 in OpenSky database; cache the miss too
            self._metadata_cache[key] = None
            return None
        except Exception as exc:
            logger.debug(f"Metadata lookup failed for {icao24}: {exc}")
            return None

    def enrich_with_type(self, flights: List[Dict],
                         max_lookups: int = 40) -> List[Dict]:
        """
        For non-civilian candidate aircraft, fetch aircraft type metadata and
        boost the classification score if the type is in TRACKED_AIRCRAFT_TYPES.

        Only performs network calls for aircraft not already in the cache.
        Caps at `max_lookups` new HTTP calls per run to stay within rate limits.
        """
        candidates = [f for f in flights if f.get('classification') != 'likely_civilian']
        if not candidates:
            return flights

        # Sort by score descending so highest-value candidates get looked up first
        candidates_sorted = sorted(
            candidates, key=lambda x: x.get('classification_score', 0), reverse=True
        )

        new_calls = 0
        for flight in candidates_sorted:
            icao24 = (flight.get('icao24') or '').lower().strip()
            if not icao24:
                continue

            cache_hit = icao24 in self._metadata_cache
            if not cache_hit and new_calls >= max_lookups:
                continue

            meta = self.fetch_aircraft_metadata(icao24)
            if not cache_hit:
                new_calls += 1
                time.sleep(0.3)   # be polite; cache will absorb repeat lookups

            if not meta:
                flight.setdefault('aircraft_type', '')
                flight.setdefault('aircraft_model', '')
                continue

            typecode = (meta.get('typecode') or '').upper()
            model    = (meta.get('model') or meta.get('manufacturername') or '').strip()
            flight['aircraft_type']  = typecode
            flight['aircraft_model'] = model
            flight['registration']   = meta.get('registration', '')
            flight['operator']       = meta.get('operatorcallsign') or meta.get('operator') or ''

            type_info = Config.TRACKED_AIRCRAFT_TYPES.get(typecode)
            if not type_info:
                continue

            boost     = type_info['weight_boost']
            old_score = flight.get('classification_score', 0)
            new_score = min(100, old_score + boost)
            flight['classification_score'] = new_score

            type_tag = f"aircraft_type:{typecode}({type_info['name']},boost=+{boost})"
            for key in ('evidence_flags', 'military_reason'):
                prev = flight.get(key, 'none')
                flight[key] = f"{prev} | {type_tag}" if prev != 'none' else type_tag

            # Re-classify if the boost crossed a threshold
            thr = Config.CONFIDENCE_THRESHOLDS
            if new_score >= thr['likely_military']:
                flight['classification'] = 'likely_military'
                flight['confidence_band'] = 'high' if new_score >= 75 else 'medium'
            elif new_score >= thr['gov_logistics'] and 'gov_logistics_callsign' in (flight.get('evidence_flags') or ''):
                flight['classification'] = 'gov_logistics'

            logger.info(
                f"Type match {icao24.upper()} → {typecode} ({type_info['name']}) "
                f"score {old_score} → {new_score}  [{flight['classification']}]"
            )

        if new_calls:
            self._save_metadata_cache()
            logger.info(f"Metadata enrichment: {new_calls} new lookups, "
                        f"{len(candidates) - new_calls} from cache")

        return flights

    def close(self):
        self._save_metadata_cache()
        self.session.close()