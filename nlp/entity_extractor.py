import re


KNOWN_ENTITIES = {

    "TCS": "TCS",
    "Tata Consultancy Services": "TCS",

    "Infosys": "Infosys",
    "INFY": "Infosys",

    "Reliance": "Reliance",
    "Reliance Industries": "Reliance",
    "Jio": "Jio",
    "JioHotstar": "JioHotstar",

    "Microsoft": "Microsoft",
    "Google": "Google",
    "Amazon": "Amazon",

    "Nvidia": "Nvidia",
    "NVIDIA": "Nvidia",

    "Intel": "Intel",
    "TSMC": "TSMC",
    "IBM": "IBM",
    "Accenture": "Accenture",
    "Oracle": "Oracle",
    "SAP": "SAP",
    "Adobe": "Adobe",
    "OpenAI": "OpenAI",
    "Meta": "Meta",
    "Apple": "Apple",

    "Kering": "Kering",
    "Porsche": "Porsche",
    "Rezolve": "Rezolve",
    "RZLV": "Rezolve",
    "Hexaware": "Hexaware",
    "Rolls-Royce": "Rolls-Royce",
    "Crocs": "Crocs",
    "SKIMS": "SKIMS",
    "Kim Kardashian": "Kim Kardashian"
}


def extract_entities(text):

    text = str(text)

    found = set()

    for entity, canonical_name in KNOWN_ENTITIES.items():

        pattern = r"\b" + re.escape(entity) + r"\b"

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            found.add(canonical_name)

    return ", ".join(
        sorted(found)
    )


def add_entities(df):

    df = df.copy()

    df["Entities"] = df["Clean_Text"].apply(
        extract_entities
    )

    return df