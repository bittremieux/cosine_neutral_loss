import numpy as np

def get_as_mods():
    as_mods = {
        "glycine": 75.032028402-18.010564683,
        "alanine": 89.047678466-18.010564683,
        "tauro": 125.01466426-18.010564683,
        "phenylalanine": 165.078978594-18.010564683,
        "tyrosine": 181.07389321-18.010564683
    }
    return np.fromiter(as_mods.values(), dtype=float)

def get_as_exchange():
    as_exchange = {
        "tauro_glycine": 125.01466426-75.032028402,
        "tauro_alanine": 125.01466426-89.047678466,
        "tauro_phenylalanine": 165.078978594-125.01466426,
        "tauro_tyrosine": 181.07389321-125.01466426,
        "glycine_phenylalanine": 165.078978594-75.032028402,
        "glycine_tyrosine": 181.07389321-75.032028402,
        "alanine_phenylalanine": 165.078978594-89.047678466,
        "alanine_tyrosine": 181.07389321-89.047678466,
    }
    return np.fromiter(as_exchange.values(), dtype=float)
