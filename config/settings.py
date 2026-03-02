import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPHA_VANTAGE_API_KEY = os.getenv('-')
    OPENSKY_USERNAME = os.getenv('-')
    OPENSKY_PASSWORD = os.getenv('-')

    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///military_oil_predictor.db')

    OPENSKY_BASE_URL = "https://opensky-network.org/api"
    REQUEST_DELAY = 2 
    MAX_RETRIES = 3
 
    OIL_SYMBOLS = ['BZ=F', 'CL=F']

    MILITARY_CALLSIGNS = [
        'USAF',     # US Air Force
        'ARMY',     # US Army
        'NAVY',     # US Navy
        'RCH',      # Reach
        'CNV',      # Convoy
        'USMC',     # US Marine Corps
        'USCG',     # US Coast Guard
        'AFO',      # Air Force One callsign prefix
        'SAM',      # Special Air Mission
        'GLEX'      # Global Express
    ]

    MILITARY_ICAO_CODES = [
        'AE',       # US Air Force
        'AF',       # US Air Force
        'A0',       # US Army
        'A1',       # US Army
        'A2',       # US Army
        'A3',       # US Army
        'A4',       # US Army
        'A5',       # US Army
        'A6',       # US Army
        'A7',       # US Army
        'A8',       # US Army
        'A9',       # US Army
        'AA',       # US Army
        'AB',       # US Army
        'AC',       # US Army
        'AD',       # US Army
        'AE',       # US Navy
        'AN',       # US Navy
        'AO',       # US Navy
        'AP',       # US Navy
        'AQ',       # US Navy
        'AR',       # US Navy
        'AS',       # US Navy
        'AT',       # US Navy
        'AU',       # US Navy
        'AV',       # US Navy
        'AW',       # US Navy
        'AX',       # US Navy
        'AY',       # US Navy
        'AZ'        # US Navy
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