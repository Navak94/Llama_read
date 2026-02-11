import pandas as pd
import re, unicodedata

hdi = pd.read_csv("HDI_2025.csv")
gpi = pd.read_csv("Global_peace_Index.csv")

def norm_key(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove accents
    s = s.replace("&", "and").replace("(", " ").replace(")", " ")
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

hdi["merge_key"] = hdi["Country"].map(norm_key)
gpi["merge_key"] = gpi["country"].map(norm_key)

# fix common naming differences (HDI -> GPI naming)
key_fix = {
    "united states": "united states of america",
    "russian federation": "russia",
    "korea republic of": "south korea",
    "iran islamic republic of": "iran",
    "viet nam": "vietnam",
    "lao peoples democratic republic": "laos",
    "syrian arab republic": "syria",
    "gambia": "the gambia",
    "congo": "republic of the congo",
    "congo democratic republic of the": "democratic republic of the congo",
}
hdi["merge_key"] = hdi["merge_key"].replace(key_fix)

merged = hdi.merge(
    gpi[["merge_key", "GPI Score"]].rename(columns={"GPI Score": "Global_Peace_Index"}),
    on="merge_key",
    how="left",
    validate="m:1"  # errors if GPI has duplicate countries
).drop(columns=["merge_key"])

merged.to_csv("HDI_2025_with_GPI.csv", index=False)
