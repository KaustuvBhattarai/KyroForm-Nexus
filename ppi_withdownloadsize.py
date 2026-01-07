import requests
import zipfile
import io
import pandas as pd
from tqdm import tqdm
import os

url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"

print("Fetching file size...")
head_response = requests.head(url, allow_redirects=True)
if 'content-length' in head_response.headers:
    total_size = int(head_response.headers['content-length'])
    size_mb = total_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
else:
    print("Could not retrieve file size (server doesn't provide it). Estimate ~400-500 MB.")
    total_size = 0

confirm = input("Proceed with download? (y/n): ").strip().lower()
if confirm != 'y':
    print("Download cancelled.")
    exit()

print("Starting download")
response = requests.get(url, stream=True)
response.raise_for_status()

# Use total_size if available, else chunk-based
chunk_size = 1024 * 1024  # 1MB chunks
progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

with open("intact.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            f.write(chunk)
            progress_bar.update(len(chunk))

progress_bar.close()
print("Downloaded N Saved as intact.zip")

print("Extracting zipzip...")
with zipfile.ZipFile("intact.zip") as z:
    file_name = [f for f in z.namelist() if f.endswith('.txt')][0]
    print(f"Extracting {file_name}...")
    z.extract(file_name)

print("Loading and filtering for human-bacteria PPIs ")
df = pd.read_csv(file_name, sep='\t', header=None, low_memory=False)

# Standard MITAB columns: 9 and 10 are taxid for A and B
df.columns = [f"col{i}" for i in range(len(df.columns))]

human_tax = "taxid:9606"
bacteria_prefix = "taxid:2"

mask = (
    ((df['col9'] == human_tax) & (df['col10'].str.startswith(bacteria_prefix, na=False))) |
    ((df['col10'] == human_tax) & (df['col9'].str.startswith(bacteria_prefix, na=False)))
)

filtered = df[mask]
output_file = "human_bacteria_ppis_intact.txt"
filtered.to_csv(output_file, sep='\t', index=False)

print(f"Done we found {len(filtered)} human bacteria interactions.")
print(f"Saved to {output_file}")

#remind me to delete the file later poorguynostorage 
