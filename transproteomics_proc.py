
import GEOparse
import logging
import os

GSE_ID = "GSE164457"
MAX_SAMPLES = 100
OUTDIR = "GEO_PROCESSED_100"

os.makedirs(OUTDIR, exist_ok=True)

logging.getLogger("GEOparse").setLevel(logging.ERROR) #toprevwarning bylord

print(f"Loadi {GSE_ID} metadata")
gse = GEOparse.get_GEO(GSE_ID)

print(f"Total GSM samples: {len(gse.gsms)}")

gsm_ids = list(gse.gsms.keys())[:MAX_SAMPLES] #limit 100? edit maxsamples

print(f"Downloading data for {len(gsm_ids)} samples")

for gsm_id in gsm_ids:
    gsm = gse.gsms[gsm_id]
    print(f"Downloading data for {gsm_id}")
    gsm.download_supplementary_files(directory=OUTDIR)

print("download complete")
