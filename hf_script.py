import pandas as pd
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm
import json
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# CONFIGURATION
# -------------------------
CSV_PATH = "Master_Inventory - Master_Inventory.csv"
OUTPUT_PATH = "Master_Inventory_enriched_ollama_optimized.csv"
OLLAMA_MODEL = "qwen2.5:7b"  
MAX_WORKERS = 2     
SAVE_INTERVAL = 5

# -------------------------
# PROMPT
# -------------------------
EXTRACTION_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior electronics engineer and technical SEO specialist. 
Extract specs and write a 50-70 word direct-answer summary for Google SEO.

<|start_header_id|>user<|end_header_id|>
Product: {part_number} - {product_name}
Datasheet: {datasheet_text}

JSON Format:
{{
  "seo_description": "Direct Answer...",
  "technical_specifications": {{
    "Part Number": "{part_number}",
    "Manufacturer": "maker",
    "Operating Voltage": "voltage",
    "Current Rating": "current",
    "Package Type": "package",
    "Key Features": ["feat1", "feat2"],
    "Applications": ["app1", "app2"]
  }}
}}
<|start_header_id|>assistant<|end_header_id|>
"""

def extract_text_from_url(url):
    if pd.isna(url) or not url or url == "Not Found":
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        if ".pdf" in url.lower():
            response = requests.get(url, timeout=15, headers=headers)
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = "\n".join([p.extract_text() for p in pdf.pages[:5] if p.extract_text()])
                return " ".join(text.split()[:2000])
        else:
            response = requests.get(url, timeout=15, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            return " ".join(soup.get_text().split()[:2000])
    except Exception:
        return None

def call_ollama(part_number, product_name, category, datasheet_text):
    try:
        prompt = EXTRACTION_PROMPT.format(
            part_number=part_number,
            product_name=product_name,
            datasheet_text=datasheet_text
        )
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, format="json", options={"temperature": 0.2, "num_ctx": 4096})
        data = json.loads(response['response'])
        return {
            "seo_description": data.get("seo_description", "N/A"),
            "technical_specifications": json.dumps(data.get("technical_specifications", {}), indent=2)
        }
    except Exception:
        return None

def process_single_row(idx, row):
    part_id = row.get("Product ID", f"ROW-{idx}")
    url = row.get("Datasheet URL", "")
    
    text = extract_text_from_url(url)
    if not text: return idx, None, None, "Extraction Failed"

    res = call_ollama(part_id, row.get("Product Name", ""), row.get("Category", ""), text)
    if not res: return idx, None, None, "AI Processing Failed"

    return idx, res["seo_description"], res["technical_specifications"], None

def main():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error: Could not find {CSV_PATH}"); return

    if "seo_description" not in df.columns: df["seo_description"] = ""
    if "technical_specifications" not in df.columns: df["technical_specifications"] = ""

    todo = df[df["seo_description"].str.len() < 10]
    total_todo = len(todo)
    
    success_count = 0
    fail_count = 0
    failed_items = []

    print(f"🚀 Starting processing for {total_todo} items...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_row, i, r): i for i, r in todo.iterrows()}
        
        # tqdm creates the progress bar
        with tqdm(total=total_todo, desc="Enriching Inventory", unit="part") as pbar:
            for future in as_completed(futures):
                idx, seo, specs, err = future.result()
                part_id = df.at[idx, "Product ID"]
                
                if not err:
                    df.at[idx, "seo_description"] = seo
                    df.at[idx, "technical_specifications"] = specs
                    success_count += 1
                else:
                    fail_count += 1
                    failed_items.append(f"{part_id}: {err}")
                
                # Update progress bar suffix with current stats
                pbar.set_postfix({"✅ Success": success_count, "❌ Fail": fail_count})
                pbar.update(1)

                if (success_count + fail_count) % SAVE_INTERVAL == 0:
                    df.to_csv(OUTPUT_PATH, index=False)

    # Final Export
    df.to_csv(OUTPUT_PATH, index=False)

    # Final Summary Report
    print("\n" + "="*40)
    print("📊 FINAL PROCESSING REPORT")
    print("="*40)
    print(f"✅ Successfully Processed: {success_count}")
    print(f"❌ Failed to Process:     {fail_count}")
    print(f"💾 File Saved to:         {OUTPUT_PATH}")
    
    if failed_items:
        print("\n⚠️  FAILURE LOG:")
        for item in failed_items[:10]: # Show first 10 failures
            print(f"  - {item}")
        if len(failed_items) > 10:
            print(f"  ... and {len(failed_items) - 10} more.")
    print("="*40)

if __name__ == "__main__":
    main()