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
    65: "Woodwind", # Soprano Sax
    66: "Woodwind", # Alto Sax
    67: "Woodwind", # Tenor Sax
    68: "Woodwind", # Baritone Sax
    69: "Woodwind", # Oboe
    70: "Woodwind", # English Horn
    71: "Woodwind", # Bassoon
    72: "Woodwind", # Clarinet
    73: "Flute", # Piccolo
    74: "Flute", # Flute
    75: "Flute", # Recorder
    76: "Flute", # Pan Flute
    77: "Flute", # Blown Bottle
    78: "Flute", # Shakuhachi
    79: "Flute", # Whistle
    80: "Flute", # Ocarina
    81: "Synth Lead/Pad", # Lead 1 (square)
    82: "Synth Lead/Pad", # Lead 2 (sawtooth)
    83: "Synth Lead/Pad", # Lead 3 (calliope)
    84: "Synth Lead/Pad", # Lead 4 (chiff)
    85: "Synth Lead/Pad", # Lead 5 (charang)
    86: "Synth Lead/Pad", # Lead 6 (voice)
    87: "Synth Lead/Pad", # Lead 7 (fifths)
    88: "Synth Lead/Pad", # Lead 8 (bass + lead)
    89: "Synth Lead/Pad", # Pad 1 (new age)
    90: "Synth Lead/Pad", # Pad 2 (warm)
    91: "Synth Lead/Pad", # Pad 3 (polysynth)
    92: "Synth Lead/Pad", # Pad 4 (choir)
    93: "Synth Lead/Pad", # Pad 5 (bowed)
    94: "Synth Lead/Pad", # Pad 6 (metallic)
    95: "Synth Lead/Pad", # Pad 7 (halo)
    96: "Synth Lead/Pad", # Pad 8 (sweep)
    97: "Synth", # FX 1 (rain)
    98: "Synth", # FX 2 (soundtrack)
    99: "Synth", # FX 3 (crystal)
    100: "Synth", # FX 4 (atmosphere)
    101: "Sytn", # FX 5 (brightness)
    102: "Synth", # FX 6 (goblins)
    103: "Synth", # FX 7 (echoes)
    104: "Synth", # FX 8 (sci-fi)
    105: "World Instrument", # Sitar
    106: "World Instrument", # Banjo
    107: "World Instrument", # Shamisen
    108: "World Instrument", # Koto
    109: "World Instrument", # Kalimba
    110: "World Instrument", # Bag pipe
    111: "World Instrument", # Fiddle
    112: "World Instrument", # Shanai
    113: "Percussion", # Tinkle Bell
    114: "Percussion", # Agogo
    115: "Percussion", # Steel Drums
    116: "Percussion", # Woodblock
    117: "Percussion", # Taiko Drum
    118: "Percussion", # Melodic Tom
    119: "Synth", # Synth Drum
    120: "Percussion", # Reverse Cymbal
    121: "Guitar", # Guitar Fret
    122: "Vocals", # Breath Noise
    123: "Percussion", # Seashore
    124: "Percussion", # Bird Tweet
    125: "Percussion", # Telephone Ring
    126: "Percussion", # Helicopter
    127: "Percussion", # Applause
    128: "Percussion" # Gunshot
}