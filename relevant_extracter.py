import pandas as pd
import os
from tqdm import tqdm

MAX_ROWS = 100  # Limit total output rows

# Common gut-relevant bacterial taxids (expanded a bit for better hits)
GUT_BACT_TAXIDS = {'816', '817', '818', '820', '253936', '471472', '198217', '165179', '239934'}  
# Bacteroides spp., Faecalibacterium prausnitzii, etc.

def extract_limited_ppis(input_file='intact.txt', output_dir='ppi_data', max_rows=MAX_ROWS):
    # Resolve full path in script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_dir, input_file)
    
    if not os.path.exists(full_input_path):
        print(f"Error: '{input_file}' not found in script directory:")
        print(f"   {script_dir}")
        print("Place 'intact.txt' in the same folder as this script and run again.")
        return
    
    print(f"Found file: {full_input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'limited_100_human_bacteria_ppis_intact.mitab')
    
    human_tax = "taxid:9606"
    bacteria_prefix = "taxid:2"
    
    extracted_rows = []
    gut_count = 0
    general_hb_count = 0
    
    print("Processing file in chunks (memory-safe)...")
    chunk_size = 10000
    progress = tqdm(desc="Scanning chunks", unit="chunk")
    
    with pd.read_csv(full_input_path, sep='\t', header=None, chunksize=chunk_size,
                     low_memory=True, on_bad_lines='skip', encoding='utf-8') as reader:
        
        for chunk in reader:
            if len(extracted_rows) >= max_rows:
                break
                
            chunk.columns = [f"col{i}" for i in range(len(chunk.columns))]
            
            # Filter human-bacteria interactions
            mask = (
                ((chunk['col9'] == human_tax) & chunk['col10'].str.startswith(bacteria_prefix, na=False)) |
                ((chunk['col10'] == human_tax) & chunk['col9'].str.startswith(bacteria_prefix, na=False))
            )
            hb_chunk = chunk[mask]
            
            if hb_chunk.empty:
                progress.update(1)
                continue
            
            # Extract bacterial taxid
            def get_bact_taxid(row):
                if row['col9'].startswith(bacteria_prefix):
                    return row['col9'].split(':')[-1].strip()
                elif row['col10'].startswith(bacteria_prefix):
                    return row['col10'].split(':')[-1].strip()
                return None
            
            hb_chunk = hb_chunk.copy()
            hb_chunk['bact_taxid'] = hb_chunk.apply(get_bact_taxid, axis=1)
            
            # Priority 1: Gut-relevant
            gut_mask = hb_chunk['bact_taxid'].isin(GUT_BACT_TAXIDS)
            gut_new = hb_chunk[gut_mask]
            
            # Priority 2: Any human-bacteria
            general_new = hb_chunk[~gut_mask]
            
            # Add gut ones first
            if not gut_new.empty:
                add_count = min(len(gut_new), max_rows - len(extracted_rows))
                extracted_rows.extend(gut_new.head(add_count).values.tolist())
                gut_count += add_count
            
            # Then fill with general if needed
            if len(extracted_rows) < max_rows and not general_new.empty:
                remaining = max_rows - len(extracted_rows)
                add_count = min(len(general_new), remaining)
                extracted_rows.extend(general_new.head(add_count).values.tolist())
                general_hb_count += add_count
            
            progress.update(1)
    
    progress.close()
    
    if not extracted_rows:
        print("No human-bacteria PPIs found in the file.")
        return
    
    # Save as MITAB (tab-separated, no header/index)
    result_df = pd.DataFrame(extracted_rows)
    result_df.to_csv(output_file, sep='\t', header=False, index=False)
    
    print(f"\n=== DONE ===")
    print(f"Total extracted: {len(extracted_rows)} rows (limited to {max_rows})")
    print(f"  - Gut-relevant bacteria: {gut_count}")
    print(f"  - General human-bacteria: {general_hb_count}")
    print(f"Saved to: {output_file}")
    
    # Show preview
    print("\nPreview (first 5 rows - key columns):")
    preview = pd.read_csv(output_file, sep='\t', header=None, nrows=5)
    print(preview[[0, 1, 9, 10]].to_string(index=False, header=['ID_A', 'ID_B', 'Tax_A', 'Tax_B']))

if __name__ == "__main__":
    extract_limited_ppis()