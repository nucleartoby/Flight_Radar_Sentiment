import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # FlightRadar24 credentials
    FR24_EMAIL    = os.getenv('FR24_EMAIL', '')
    FR24_PASSWORD = os.getenv('FR24_PASSWORD', '')

    REQUEST_DELAY = 2
    MAX_RETRIES   = 3

    OIL_SYMBOLS = ['BZ=F', 'CL=F']

    STRONG_MIL_CALLSIGNS = [
        'USAF', 'ARMY', 'NAVY', 'USMC', 'USCG',
        'AFO',                   # Air Force One prefix
        'SAM',                   # Special Air Mission
        'CNV', 'CONVOY',
        'MAGIC',                 # AWACS
        'ASCOT',                 # UK military air transport (RAF telephony)
        'RRR',                   # RAF ICAO designator (telephony ASCOT)
        'RFR',                   # RAF ICAO designator (telephony RAFAIR)
        'TARTAN',                # RAF Scotland QRA
        'COTAM',                 # French Air Force transport command
        'FAF',                   # French Air Force
        'NATO',                  # NATO declared callsign
        'TUAF', 'TUF',           # Turkish Air Force
        # USAF airlift/tanker tactical callsigns observed in the AOI
        'MOOSE',                 # C-17 airlift
        'CADDY',                 # US military light transport
        'BLUE',                  # tanker/airlift tactical
        'QID', 'QUID',           # KC-135 tactical
        'ETHYL',                 # tanker tactical
        'TABOR', 'CHROME',       # tanker tactical
    ]

    GOV_LOGISTICS_CALLSIGNS = [
        'CMB',                   # Camber
        'RCH', 'REACH',          # Air Mobility Command (telephony REACH)
        'SHELL',                 # KC tanker tactical callsign
        'GOTHAM',                # ISR platform
        'VENUS', 'BART',         # ISR / surveillance
        'GLEX',                  # Global Express
    ]

    WEAK_GOV_CALLSIGNS = [
        'EXEC',
        'VIP',
    ]

    COMMERCIAL_CALLSIGN_PREFIXES = [
        # Gulf carriers
        'UAE', 'ETD', 'QTR', 'GFA', 'BAH', 'OMA', 'FDB',
        # Middle East / Levant
        'SVA', 'MSR', 'RJA', 'ELY', 'MEA', 'IAW', 'IRA',
        # Turkish (commercial)
        'THY', 'PGT', 'TRK',
        # European majors transiting
        'BAW', 'AAL', 'DAL', 'UAL', 'AFR', 'DLH', 'KLM', 'IBE', 'VIR', 'SIA', 'CPA',
        # Budget / charter
        'WZZ', 'RYR', 'EZY', 'EXS',
        # South Asian
        'AIC', 'IGO', 'PIA',
        # African
        'ETH', 'KQA',
    ]

    # FR24 provides airline_iata directly — use as a strong negative signal.
    # These are IATA airline codes (2-letter), not callsign prefixes.
    COMMERCIAL_AIRLINE_IATA = {
        'EK', 'EY', 'QR', 'GF', 'WY', 'G9', 'FZ',  # Gulf carriers
        'SV', 'ME', 'RJ', 'LY', 'MS', 'IA', 'IR',   # Middle East / Levant
        'TK', 'PC', 'XQ',                             # Turkish
        'BA', 'AF', 'LH', 'KL', 'IB', 'VS',          # European majors
        'W6', 'FR', 'U2',                             # Low-cost
        'AI', '6E', 'PK', 'ET',                      # Other major
    }

    # strong_mil must be >= CONFIDENCE_THRESHOLDS['likely_military'] so that a
    # confirmed military callsign is sufficient on its own. At the old value of
    # 55 an RAF Voyager squawking RRR scored below the 65 threshold and fell
    # through to 'unknown'.
    CALLSIGN_WEIGHTS = {
        'strong_mil':    70,
        'gov_logistics': 45,
        'weak_gov':      12,
    }

    CATEGORY_WEIGHTS = {
        7:  35,
        14: 30,
        8:  12,
        0:   8,
    }

    # Replaces the old ICAO_BLOCK_WEIGHTS 2-char prefix table, which was wrong in
    # two ways: 'A0'-'AD' is the US *civil* block (a00001=N1 .. adf7c7=N99999, so
    # it rewarded every US airliner in frame), and 'AN'-'AZ' contained non-hex
    # characters that could never match. Full-range lookup now lives in
    # src/data/mil_hex.py against the vendored config/mil_ranges.json.
    #
    # Weight is >= the likely_military threshold: a hit in a military allocation
    # block is the single strongest indicator available in an ADS-B feed and is
    # sufficient on its own. A miss is NOT evidence of civilian status -- see the
    # spoofing note in src/data/mil_hex.py.
    MIL_HEX_WEIGHT = 65

    BASE_TYPE_WEIGHTS = {
        'isr_hub':       20,
        'joint_base':    16,
        'air_base':      14,
        'naval_base':    10,
        'logistics_hub':  8,
        'garrison':       8,
    }

    NEGATIVE_WEIGHTS = {
        'commercial_prefix': -50,
        'iata_format':       -28,
    }

    CONFIDENCE_THRESHOLDS = {
        'likely_military': 65,
        'gov_logistics':   45,
        'unknown':         25,
    }

    MILITARY_CALLSIGNS = STRONG_MIL_CALLSIGNS + GOV_LOGISTICS_CALLSIGNS + WEAK_GOV_CALLSIGNS

    TRACKED_AIRCRAFT_TYPES = {
        # Tankers
        'K35R':  {'name': 'KC-135R Stratotanker',    'weight_boost': 30},
        'KC46':  {'name': 'KC-46A Pegasus',           'weight_boost': 30},
        # Strategic / tactical airlift
        'C17':   {'name': 'C-17 Globemaster III',     'weight_boost': 22},
        'C130':  {'name': 'C-130 Hercules',            'weight_boost': 18},
        'C13J':  {'name': 'C-130J Super Hercules',    'weight_boost': 18},
        'C5M':   {'name': 'C-5M Super Galaxy',        'weight_boost': 22},
        # ISR / surveillance
        'U2':    {'name': 'Lockheed U-2',              'weight_boost': 45},
        'RQ4B':  {'name': 'RQ-4B Global Hawk',        'weight_boost': 45},
        'MQ9':   {'name': 'MQ-9 Reaper',              'weight_boost': 40},
        'P8':    {'name': 'P-8A Poseidon',            'weight_boost': 35},
        'E3TF':  {'name': 'E-3 Sentry AWACS',         'weight_boost': 45},
        'R135':  {'name': 'RC-135 (recon variant)',   'weight_boost': 45},
        # Strike / multirole fighters
        'F35A':  {'name': 'F-35A Lightning II',       'weight_boost': 40},
        'F35B':  {'name': 'F-35B Lightning II',       'weight_boost': 40},
        'F35C':  {'name': 'F-35C Lightning II',       'weight_boost': 40},
        'F15E':  {'name': 'F-15E Strike Eagle',       'weight_boost': 35},
        'F15C':  {'name': 'F-15C Eagle',              'weight_boost': 35},
        'F16C':  {'name': 'F-16C Fighting Falcon',    'weight_boost': 32},
        'F16D':  {'name': 'F-16D Fighting Falcon',    'weight_boost': 32},
        'A10C':  {'name': 'A-10C Thunderbolt II',     'weight_boost': 35},
        # Bombers
        'B52':   {'name': 'B-52 Stratofortress',      'weight_boost': 50},
        'B2':    {'name': 'B-2 Spirit',               'weight_boost': 50},
        'B1B':   {'name': 'B-1B Lancer',              'weight_boost': 45},
        # Carrier-based / Navy
        'E2':    {'name': 'E-2D Advanced Hawkeye',    'weight_boost': 45},
        'E2C':   {'name': 'E-2C Hawkeye',             'weight_boost': 45},
        'EA18':  {'name': 'EA-18G Growler',           'weight_boost': 45},
        'F18S':  {'name': 'F/A-18E/F Super Hornet',   'weight_boost': 38},
        # Additional ISR
        'E8C':   {'name': 'E-8C JSTARS',              'weight_boost': 45},
        'EP3':   {'name': 'EP-3E Aries II (SIGINT)',  'weight_boost': 45},
    }

    BASE_RADIUS_KM           = 60    # global fallback when per-base radius absent
    DATA_COLLECTION_INTERVAL = 600

    PREDICTION_HORIZON_DAYS = 7
    TRAIN_TEST_SPLIT        = 0.8
    RANDOM_STATE            = 42

    LOG_LEVEL = 'INFO'
    LOG_FILE  = 'logs/military_oil_predictor.log'

    CACHE_DURATION_MINUTES = 5
    ENABLE_DATA_CACHING    = True

    # --- Lifecycle tracking (src/tracking) ---
    DB_PATH               = 'data/tracking.db'
    POLL_INTERVAL_SEC     = 60      # how often to poll the flight feed
    OIL_REFRESH_SEC       = 900     # how often to refresh intraday oil prices
    LANDING_TIMEOUT_MIN   = 15      # not seen for this long -> close event as signal_lost
    TRACK_ONLY_MILITARY   = True    # only open events for is_military aircraft
    OIL_INTRADAY_INTERVAL = '60m'   # yfinance interval for intraday oil bars
    OIL_INTRADAY_PERIOD   = '5d'    # yfinance lookback per refresh
    SNAPSHOT_SAMPLES      = 2       # quick polls per cycle, deduped by icao24
    SNAPSHOT_RETRIES      = 4       # retries when the feed returns zero aircraft
    SNAPSHOT_RETRY_WAIT   = 2       # seconds between retries
    TRACKING_LOG_FILE     = 'logs/flight_tracker.log'

    # Bumped whenever classify_flight scoring changes. Stored on every row so a
    # rescore never silently mixes regimes -- scores from different rulesets are
    # not comparable and must not be pooled into one index series.
    RULESET_VERSION = 2

    # Record aircraft that failed the is_military filter but sit in a military
    # ICAO allocation block. This is the false-negative set: without it, recall
    # is unmeasurable because rejected aircraft leave no trace.
    LOG_FALSE_NEGATIVES = True