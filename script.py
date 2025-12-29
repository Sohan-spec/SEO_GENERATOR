import pandas as pd
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# -------------------------
# CONFIGURATION
# -------------------------
CSV_PATH = "Master_Inventory - Master_Inventory.csv"
OUTPUT_PATH = "Master_Inventory_enriched.csv"

# Ollama model settings
# Recommended models (download with: ollama pull <model>):
# "llama3.2:3b" - Fast, good quality (4GB)
# "llama3.1:8b" - Better quality, slower (5GB)
# "qwen2.5:7b" - Great for technical content (4.7GB)
OLLAMA_MODEL = "llama3:8b"  # Change this to your downloaded model

REQUEST_DELAY = 0.5  # Small delay between requests (local, so fast)
MAX_WORKERS = 2  # Conservative for local processing
SAVE_INTERVAL = 5
DEBUG_MODE = True

# -------------------------
# TEXT EXTRACTION (downloads PDFs from URLs)
# -------------------------
def extract_text_from_url(url):
    """Extract text from PDF or HTML datasheet - USES INTERNET"""
    if pd.isna(url) or not url or url == "Not Found":
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Handle PDF files
        if '.pdf' in url.lower() or '/pdf/' in url.lower():
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text_parts = []
                for page in pdf.pages[:8]:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                full_text = "\n".join(text_parts)
                words = full_text.split()[:3500]  # Limit for local model
                return " ".join(words)
        
        # Handle HTML/web pages
        else:
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            words = text.split()[:3500]
            return " ".join(words)
    
    except Exception as e:
        print(f"\n⚠️  Error extracting from {url[:50]}... - {str(e)}")
        return None

# -------------------------
# PROMPT
# -------------------------
EXTRACTION_PROMPT = """You are an electronics technical expert. Extract specifications from this datasheet.

Product: {part_number} - {product_name} ({category})

Datasheet:
{datasheet_text}

Output ONLY valid JSON, no other text:

{{
  "seo_description": "120-150 word product description with key features and applications",
  "technical_specifications": {{
    "Part Number": "{part_number}",
    "Product Type": "type",
    "Manufacturer": "maker",
    "Operating Voltage": "voltage",
    "Current Rating": "current",
    "Package Type": "package",
    "Temperature Range": "temp range",
    "Interface": "interface type",
    "Key Features": ["feature1", "feature2"],
    "Applications": ["app1", "app2"]
  }}
}}

Rules: Extract only stated info. Use "N/A" if not found. Output ONLY JSON, no markdown, no explanations."""

# -------------------------
# OLLAMA API CALL (LOCAL)
# -------------------------
def call_ollama(part_number, product_name, category, datasheet_text, max_retries=2):
    
    for attempt in range(max_retries):
        try:
            prompt = EXTRACTION_PROMPT.format(
                part_number=part_number,
                product_name=product_name,
                category=category,
                datasheet_text=datasheet_text
            )
            
            # Call Ollama via subprocess with proper encoding
            process = subprocess.Popen(
                ['ollama', 'run', OLLAMA_MODEL],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # Force UTF-8 encoding
                errors='replace'   # Replace problematic characters
            )
            
            stdout, stderr = process.communicate(input=prompt, timeout=120)
            
            if process.returncode != 0:
                raise Exception(f"Ollama error: {stderr}")
            
            result_text = stdout.strip()
            
            # Clean markdown formatting
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            # Find JSON boundaries
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                result_text = result_text[start_idx:end_idx+1]
            
            result_text = result_text.strip()
            
            # Parse JSON
            data = json.loads(result_text)
            specs = data.get('technical_specifications', {})
            
            if DEBUG_MODE:
                print(f"\n✓ {part_number} - Extracted {len(specs)} specifications")
            
            specs_formatted = json.dumps(specs, indent=2, ensure_ascii=False)
            
            return {
                'seo_description': data.get('seo_description', 'N/A'),
                'technical_specifications': specs_formatted
            }
        
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parse error for {part_number}: {e}")
            if DEBUG_MODE and 'result_text' in locals():
                print(f"   Raw output (first 300 chars): {result_text[:300]}")
            if attempt < max_retries - 1:
                print(f"   Retrying...")
                time.sleep(2)
                continue
            return None
        
        except subprocess.TimeoutExpired:
            print(f"\n⚠️  Timeout for {part_number} (model took too long)")
            process.kill()
            return None
        
        except Exception as e:
            print(f"\n⚠️  Error for {part_number}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    
    return None

# -------------------------
# PROCESS SINGLE ROW
# -------------------------
def process_single_row(idx, row):
    """Process one product entry"""
    part_number = row.get('Product ID', f'ROW-{idx}')
    product_name = row.get('Product Name', 'Unknown')
    category = row.get('Category', 'Unknown')
    datasheet_url = row.get('Datasheet URL', '')
    
    try:
        # Extract datasheet text (uses internet)
        datasheet_text = extract_text_from_url(datasheet_url)
        
        if not datasheet_text or len(datasheet_text.strip()) < 200:
            return idx, None, None, f"Insufficient text extracted"
        
        # Call Ollama (local processing, no internet)
        time.sleep(REQUEST_DELAY)
        result = call_ollama(part_number, product_name, category, datasheet_text)
        
        if not result:
            return idx, None, None, "Model returned no data"
        
        return idx, result['seo_description'], result['technical_specifications'], None
    
    except Exception as e:
        return idx, None, None, str(e)

# -------------------------
# CHECK OLLAMA INSTALLATION
# -------------------------
def check_ollama():
    """Check if Ollama is installed and model is available"""
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Ollama is not installed!")
            print("\nInstall with:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            print("\nOr visit: https://ollama.com/download")
            return False
        
        # Check if model is downloaded
        if OLLAMA_MODEL not in result.stdout:
            print(f"❌ Model '{OLLAMA_MODEL}' is not downloaded!")
            print(f"\nDownload with:")
            print(f"  ollama pull {OLLAMA_MODEL}")
            print("\nAvailable models:")
            print("  ollama pull llama3.2:3b    (4GB - Fast)")
            print("  ollama pull llama3.1:8b    (5GB - Better quality)")
            print("  ollama pull qwen2.5:7b     (4.7GB - Great for technical)")
            return False
        
        print(f"✅ Ollama is ready with model: {OLLAMA_MODEL}")
        return True
    
    except FileNotFoundError:
        print("❌ Ollama is not installed!")
        print("\nInstall with:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("\nOr visit: https://ollama.com/download")
        return False

# -------------------------
# MAIN PROCESSING
# -------------------------
def main():
    print("=" * 60)
    print("Electronic Components Datasheet Enrichment")
    print(f"Provider: Ollama (Local, 100% FREE)")
    print(f"Model: {OLLAMA_MODEL}")
    print("=" * 60)
    
    # Check Ollama installation
    if not check_ollama():
        return
    
    # Load CSV
    print(f"\n📂 Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   ✓ Loaded {len(df)} rows")
    
    # Add new columns
    if 'seo_description' not in df.columns:
        df['seo_description'] = ''
    if 'technical_specifications' not in df.columns:
        df['technical_specifications'] = ''
    
    # Find rows to process
    rows_to_process = []
    for idx, row in df.iterrows():
        if not df.at[idx, 'seo_description'] or len(str(df.at[idx, 'seo_description'])) < 50:
            rows_to_process.append((idx, row))
    
    total = len(rows_to_process)
    print(f"\n🔄 Found {total} rows to process")
    
    if total == 0:
        print("   ✓ All rows already processed!")
        return
    
    # Estimate time (local models are slower)
    estimated_minutes = (total * 15) / 60  # ~15 seconds per item
    print(f"⏱️  Estimated time: ~{estimated_minutes:.0f} minutes")
    
    # Process rows
    successful = 0
    failed = 0
    
    print(f"\n⚙️  Processing with {MAX_WORKERS} parallel workers...\n")
    print("💡 Note: Local processing is slower but FREE with no limits!\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_row, idx, row): idx
            for idx, row in rows_to_process
        }
        
        with tqdm(total=total, desc="Processing", unit="product") as pbar:
            for future in as_completed(future_to_idx):
                idx, seo_desc, specs, error = future.result()
                
                if error:
                    failed += 1
                    part_id = df.at[idx, 'Product ID'] if 'Product ID' in df.columns else idx
                    tqdm.write(f"❌ {part_id}: {error}")
                else:
                    successful += 1
                    df.at[idx, 'seo_description'] = seo_desc
                    df.at[idx, 'technical_specifications'] = specs
                
                pbar.update(1)
                
                if (successful + failed) % SAVE_INTERVAL == 0:
                    df.to_csv(OUTPUT_PATH, index=False)
                    tqdm.write(f"💾 Progress saved ({successful} successful, {failed} failed)")
    
    # Final save
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"💰 Cost: FREE (Local processing)")
    print(f"📄 Output: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()