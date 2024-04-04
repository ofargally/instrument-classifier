import numpy as np
groups = {
    1: "Keyboard", # Acoustic Grand Piano
    2: "Keyboard", # Bright Acoustic Piano
    3: "Keyboard", # Electric Grand Piano
    4: "Keyboard", # Honky-tonk Piano
    5: "Keyboard", # Electric Piano 1
    6: "Keyboard", # Electric Piano 2
    7: "Keyboard", # Harpsichord
    8: "Keyboard", # Clavi
    9: "Percussion", # Celesta
    10: "Percussion", # Glockenspiel
    #11: "Music Box", # Music Box
    12: "Percussion", # Vibraphone
    13: "Percussion", # Marimba
    14: "Percussion", # Xylophone
    15: "Percussion", # Tubular Bells
    16: "Percussion", # Dulcimer
    17: "Organ", # Drawbar Organ
    18: "Organ", # Percussive Organ
    19: "Organ", # Rock Organ
    20: "Organ", # Church Organ
    21: "Organ", # Reed Organ
    #22: "Woodwind", # Accordion
    23: "Woodwind", # Harmonica
    24: "Woodwind", # Tango Accordion
    25: "Guitar", # Acoustic Guitar (nylon)
    26: "Guitar", # Acoustic Guitar (steel)
    27: "Guitar", # Electric Guitar (jazz)
    28: "Guitar", # Electric Guitar (clean)
    29: "Guitar", # Electric Guitar (muted)
    30: "Guitar", # Overdriven Guitar
    31: "Guitar", # Distortion Guitar
    32: "Guitar", # Guitar harmonics
    33: "Bass", # Acoustic Bass
    34: "Bass", # Electric Bass (finger)
    35: "Bass", # Electric Bass (pick)
    36: "Bass", # Fretless Bass
    37: "Bass", # Slap Bass 1
    38: "Bass", # Slap Bass 2
    39: "Bass", # Synth Bass 1
    40: "Bass", # Synth Bass 2
    41: "Strings", # Violin
    42: "Strings", # Viola
    43: "Strings", # Cello
    44: "Strings", # Contrabass
    45: "Strings", # Tremolo Strings
    46: "Strings", # Pizzicato Strings
    47: "Strings", # Orchestral Harp
    48: "Percussion", # Timpani
    49: "Strings", # String Ensemble 1
    50: "Strings", # String Ensemble 2
    51: "Strings", # SynthStrings 1
    52: "Strings", # SynthStrings 2
    53: "Vocals", # Choir Aahs
    54: "Vocals", # Voice Oohs
    55: "Vocals", # Synth Voice
    56: "Strings", # Orchestra Hit
    57: "Brass", # Trumpet
    58: "Brass", # Trombone
    59: "Brass", # Tuba
    60: "Brass", # Muted Trumpet
    61: "Brass", # French Horn
    62: "Brass", # Brass Section
    63: "Brass", # SynthBrass 1
    64: "Brass", # SynthBrass 2
    65: "Woodwinds", # Soprano Sax
    66: "Woodwinds", # Alto Sax
    67: "Woodwinds", # Tenor Sax
    68: "Woodwinds", # Baritone Sax
    69: "Oboe",
    70: "English Horn",
    71: "Bassoon",
    72: "Clarinet",
    73: "Piccolo",
    74: "Flute",
    75: "Recorder",
    76: "Pan Flute",
    77: "Blown Bottle",
    78: "Shakuhachi",
    79: "Whistle",
    80: "Ocarina",
    81: "Lead 1 (square)",
    82: "Lead 2 (sawtooth)",
    83: "Lead 3 (calliope)",
    84: "Lead 4 (chiff)",
    85: "Lead 5 (charang)",
    86: "Lead 6 (voice)",
    87: "Lead 7 (fifths)",
    88: "Lead 8 (bass + lead)",
    89: "Pad 1 (new age)",
    90: "Pad 2 (warm)",
    91: "Pad 3 (polysynth)",
    92: "Pad 4 (choir)",
    93: "Pad 5 (bowed)",
    94: "Pad 6 (metallic)",
    95: "Pad 7 (halo)",
    96: "Pad 8 (sweep)",
    97: "FX 1 (rain)",
    98: "FX 2 (soundtrack)",
    99: "FX 3 (crystal)",
    100: "FX 4 (atmosphere)",
    101: "FX 5 (brightness)",
    102: "FX 6 (goblins)",
    103: "FX 7 (echoes)",
    104: "FX 8 (sci-fi)",
    105: "Sitar",
    106: "Banjo",
    107: "Shamisen",
    108: "Koto",
    109: "Kalimba",
    110: "Bag pipe",
    111: "Fiddle",
    112: "Shanai",
    113: "Tinkle Bell",
    114: "Agogo",
    115: "Steel Drums",
    116: "Woodblock",
    117: "Taiko Drum",
    118: "Melodic Tom",
    119: "Synth Drum",
    120: "Reverse Cymbal",
    121: "Guitar Fret Noise",
    122: "Breath Noise",
    123: "Seashore",
    124: "Bird Tweet",
    125: "Telephone Ring",
    126: "Helicopter",
    127: "Applause",
    128: "Gunshot"
}