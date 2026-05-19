import os
import json
from dotenv import load_dotenv

load_dotenv()


def _load_credentials() -> dict:
    cred_path = os.path.join(os.path.dirname(__file__), '..', 'credentials.json')
    cred_path = os.path.normpath(cred_path)
    if os.path.exists(cred_path):
        with open(cred_path) as f:
            return json.load(f)
    return {}


_creds = _load_credentials()


class Config:
    OPENSKY_CLIENT_ID     = _creds.get('clientId',     os.getenv('OPENSKY_CLIENT_ID', ''))
    OPENSKY_CLIENT_SECRET = _creds.get('clientSecret', os.getenv('OPENSKY_CLIENT_SECRET', ''))

    OPENSKY_USERNAME = os.getenv('OPENSKY_USERNAME', '')
    OPENSKY_PASSWORD = os.getenv('OPENSKY_PASSWORD', '')

    OPENSKY_TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/opensky-network"
        "/protocol/openid-connect/token")
    
    OPENSKY_BASE_URL = "https://opensky-network.org/api"

    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///military_oil_predictor.db')

    REQUEST_DELAY = 2
    MAX_RETRIES = 3

    OIL_SYMBOLS = ['BZ=F', 'CL=F']

    MILITARY_CALLSIGNS = [
        # US military
        'USAF', 'ARMY', 'NAVY', 'USMC', 'USCG',
        'RCH', 'REACH',           # Air Mobility Command
        'SAM',                    # Special Air Mission (VIP/exec)
        'AFO',                    # Air Force One prefix
        'GLEX',                   # Global Express (often mil/gov)
        'CNV', 'CONVOY',
        'MAGIC',                  # AWACS
        'SHELL',                  # tanker
        'GOTHAM',                 # ISR
        'VENUS', 'BART',          # ISR/surveillance
        # UK military
        'ASCOT', 'TARTAN', 'RRR',
        # French military
        'COTAM', 'FAF',
        # NATO / generic
        'NATO',
        # Turkish military
        'TUF', 'TUAF',
        # Government / state (non-military but monitored)
        'EXEC',   # US government executive flight
        'VIP',
    ]

    MILITARY_ICAO_CODES = [
        'AE', 'AF',
        'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
        'AA', 'AB', 'AC', 'AD',
        'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
    ]

    BASE_RADIUS_KM = 100
    DATA_COLLECTION_INTERVAL = 600

    PREDICTION_HORIZON_DAYS = 7
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 42

    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/military_oil_predictor.log'

    CACHE_DURATION_MINUTES = 5
    ENABLE_DATA_CACHING = True
