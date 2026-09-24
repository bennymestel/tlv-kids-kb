"""Words from the chat that search misses -> plain English, for query expansion.

Covers transliterated Hebrew (mazgan, moked) and English shorthand (AC, JLM, GP),
since the saved questions spell things out in full English.
If a search query contains one of these words, its translation is appended to the
query (the original word is kept), so the embedding search understands it.
Values are plain translations of the word, not guesses at what the user wants.
Spelling variants are listed as separate keys. Keys may be multi-word phrases.
"""
import re

TERMS = {
    # home & repairs
    "mazgan": "air conditioner AC",
    "mazganim": "air conditioners AC",
    "musach": "car garage",
    "mosach": "car garage",
    "shiputznik": "renovation contractor",
    "shiputz": "renovation",
    "instalator": "plumber",
    "installator": "plumber",
    "chashmalai": "electrician",
    "hashmalai": "electrician",
    "manulan": "locksmith",
    "dud shemesh": "solar water heater",
    "dud": "water heater boiler",
    "mamad": "safe room",
    "miklat": "bomb shelter",
    "miklats": "bomb shelters",
    "mirpeset": "balcony",
    "machsan": "storage room",
    "vaad bayit": "building committee",
    "pinui binui": "urban renewal",
    "dirota": "apartments",
    "kumkum": "electric kettle",
    "plata": "hot plate",
    "jukim": "cockroaches",
    "juk": "cockroach",
    "hadbara": "pest extermination",
    "madbir": "exterminator",
    # health
    "moked": "urgent care center",
    "mokad": "urgent care center",
    "kupat cholim": "health fund",
    "kupah": "health fund",
    "tipat chalav": "baby wellness clinic",
    "metapelet": "nanny",
    "rofe": "doctor",
    "rofa": "doctor",
    # bureaucracy & services
    "arnona": "municipal property tax",
    "iriya": "municipality",
    "iriyah": "municipality",
    "irya": "municipality",
    "misrad hapnim": "ministry of interior",
    "misrad": "office ministry",
    "teudat zehut": "ID card",
    "darkon": "passport",
    "osek patur": "exempt small business",
    "osek murshe": "licensed business",
    "osek": "business",
    "bituach leumi": "national insurance",
    "tofer": "tailor",
    "toferet": "seamstress",
    "sapar": "barber",
    "mispara": "hair salon",
    "madim": "uniforms",
    "chayal": "soldier",
    "miluim": "army reserve duty",
    "miluimnik": "army reservist",
    "tironut": "basic training",
    "gemach": "free loan fund",
    "ulpan": "Hebrew language school",
    "olim": "new immigrants",
    "oleh": "new immigrant",
    "olah": "new immigrant",
    "aliyah": "immigration to Israel",
    "aliya": "immigration to Israel",
    "gan": "kindergarten",
    "kaytana": "day camp",
    "kaitana": "day camp",
    # transport
    "hasaot": "transportation shuttles",
    "hasaa": "ride",
    "moreh nehiga": "driving instructor",
    "nehiga": "driving",
    "sherut": "shared taxi",
    "monit": "taxi",
    "tachana": "station",
    # food & shopping
    "shuk": "market",
    "makolet": "grocery store",
    "sufganiot": "doughnuts",
    "sufganiyot": "doughnuts",
    "sufganiya": "doughnut",
    "jachnun": "Yemenite pastry",
    "malawach": "Yemenite fried bread",
    "burekas": "savory pastry",
    "bourekas": "savory pastry",
    "sabich": "eggplant pita sandwich",
    "kashrut": "kosher certification",
    "hashgacha": "kosher supervision",
    "mehadrin": "strictly kosher",
    # religious
    "mikveh": "ritual bath",
    "mikvah": "ritual bath",
    "mikva": "ritual bath",
    "mikve": "ritual bath",
    "kelim": "dishes",
    "keilim": "dishes",
    "tevila": "ritual immersion",
    "tevilah": "ritual immersion",
    "tallit": "prayer shawl",
    "tallis": "prayer shawl",
    "talit": "prayer shawl",
    "kippa": "skullcap",
    "kippah": "skullcap",
    "tzitzit": "ritual fringes",
    "rabanut": "rabbinate",
    "rav": "rabbi",
    "tayelet": "promenade",
    # common English shorthand
    "ac": "air conditioner",
    "a/c": "air conditioner",
    "jlm": "Jerusalem",
    "ent": "ear nose throat doctor",
    "gp": "family doctor",
    "derm": "dermatologist",
    "physio": "physiotherapist",
    "dr": "doctor",
    "doc": "doctor",
    "apt": "apartment",
    "apts": "apartments",
    "bday": "birthday",
    "mda": "ambulance service",
    "idf": "army",
    "nis": "shekels",
    "ss": "Shabbat observant",
    "sk": "kosher kitchen",
}

# Longest keys first so "dud shemesh" wins over "dud", "moreh nehiga" over "nehiga".
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand(query: str) -> str:
    """'mazgan guy' -> 'mazgan guy air conditioner AC'."""
    glosses = []
    for m in _PATTERN.finditer(query):
        gloss = TERMS[m.group(1).lower()]
        if gloss not in glosses:
            glosses.append(gloss)
    return " ".join([query] + glosses)
