import pandas as pd
import GEOparse

gse = GEOparse.get_GEO(filepath="GSE164457_family.soft.gz")

# Convert sample metadata to DataFrame
rows = []
for gsm_id, gsm in gse.gsms.items():
    row = {k: v[0] if isinstance(v, list) else v
           for k, v in gsm.metadata.items()}
    row["GSM"] = gsm_id
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("GSE164457_metadata.csv", index=False)
