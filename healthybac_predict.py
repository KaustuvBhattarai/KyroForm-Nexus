import requests
import zipfile
import os
from tqdm import tqdm

url = "https://zenodo.org/records/14780446/files/Healty_Bac_predictions.zip?download=1"
filename = "Healty_Bac_predictions.zip"

print("Fetching file size...")
head_response = requests.head(url, allow_redirects=True)
if 'content-length' in head_response.headers:
    total_size = int(head_response.headers['content-length'])
    size_mb = total_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB (~419 MB)")
else:
    print("Could not retrieve exact size from server. Known size: ~419 MB")
    total_size = None

confirm = input("\nProceed with download? (y/n): ").strip().lower()
if confirm != 'y':
    print("Download cancelled.")
    exit()

print("Starting download...")
response = requests.get(url, stream=True)
response.raise_for_status()

chunk_size = 1024 * 1024  # 1MB chunks
progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", miniters=1)

with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            f.write(chunk)
            progress_bar.update(len(chunk))

progress_bar.close()
print(f"Download complete! Saved as {filename}")

print("Extracting ZIP (this may take a few minutes)...")
with zipfile.ZipFile(filename) as z:
    z.extractall("gut_ppi_predictions_data")  # Extracts to a new folder

print("Extraction complete! Files are in ./gut_ppi_predictions_data/")
print("Inside: JSON files per human protein + tab-separated PPI list for our graph positives.")