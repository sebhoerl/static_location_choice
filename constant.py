import numpy as np

ACTIVITY_TYPES = ["work", "education", "shop", "leisure", "remote_work", "escort_kids", "escort_other", "home", "remote_home"]
MODES = ["car", "pt", "bike", "walk"]

ACTIVITY_TYPES_TO_INDEX = { v : k for k, v in enumerate(ACTIVITY_TYPES) }
MODES_TO_INDEX = { v : k for k, v in enumerate(MODES) }

PURPOSE_MAP = {
    '-99' : None,
    '1': '#umsteigen',
    '2': 'work',
    '3': 'education',
    '4': 'shop',
    '5': 'shop',
    '6': 'leisure',
    '7': 'remote_work',
    '8': 'leisure',
    '9': 'escort_kids',
    '10': 'escort_other',
    '11': 'home',
    '12': None
}

MODE_MAP = {
    '-99' : None,
    '1' : None,
    '2' : 'pt', # Bahn
    '3' : 'pt', # Postauto
    '4' : 'pt', # Schiff
    '5' : 'pt', # Tram
    '6' : 'pt', # Bus
    '7' : 'pt', # Sonstig OEV
    '8' : 'car', # Reisecar
    '9' : 'car', # Auto
    '10' : None,
    '11' : None,
    '12' : None,
    '13' : None,
    '14' : "bike", #'bike',
    '15' : "walk", #'walk',
    '16' : None,
    '17' : None
}
