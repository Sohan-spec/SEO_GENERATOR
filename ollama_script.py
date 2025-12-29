import pandas as pd
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
import re
import gc
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------
# CONFIGURATION
# -------------------------
CSV_PATH = "Master_Inventory - Master_Inventory.csv"
OUTPUT_PATH = "Master_Inventory_enriched.csv"
OLLAMA_MODEL = "qwen2.5:7b"

REQUEST_DELAY = 0.8
MAX_WORKERS = 2
SAVE_INTERVAL = 3
DEBUG_MODE = True

# -------------------------
# AGGRESSIVE TEXT EXTRACTION
# -------------------------
def extract_text_from_url(url):
    """Extract maximum text from datasheets"""
    if pd.isna(url) or not url or str(url).lower() in ["not found", "nan", ""]:
        return None
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        
        response = requests.get(
            url, 
            timeout=45, 
            headers=headers, 
            verify=False, 
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Handle PDFs - Extract MORE pages for better data
        if ".pdf" in url.lower() or "pdf" in content_type:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                # Extract first 10 pages - specs are usually in first few pages
                all_text = []
                for i, page in enumerate(pdf.pages[:10]):
                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            all_text.append(f"PAGE {i+1}:\n{text}")
                    except:
                        continue
                
                full_text = "\n\n".join(all_text)
                
                # Take first 5000 words - more data = better extraction
                if full_text:
                    words = full_text.split()[:5000]
                    return " ".join(words)
                return None
        
        # Handle HTML/Web pages
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                tag.decompose()
            
            # Try to find main content areas (common class names for product specs)
            main_content = soup.find_all(['main', 'article', 'section', 'div'], 
                                        class_=re.compile(r'(content|product|spec|detail|description|feature)', re.I))
            
            if main_content:
                text = "\n".join([elem.get_text(separator='\n', strip=True) for elem in main_content])
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n', text)
            
            # Take first 5000 words
            words = text.split()[:5000]
            return " ".join(words) if words else None
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"\n⚠️  Extraction failed for {url[:50]}... - {str(e)}")
        return None

# -------------------------
# TWO-STAGE EXTRACTION PROMPT
# -------------------------
# Stage 1: Understanding the datasheet
UNDERSTANDING_PROMPT = """Analyze this electronics datasheet text and identify key information.

Part Number: {part_number}

Datasheet Content:
{datasheet_text}

What are the most important specifications mentioned in this text? List them clearly and concisely."""

# Stage 2: Structured extraction
EXTRACTION_PROMPT = """Based on your analysis, create a structured JSON response for this electronics component.

Part Number: {part_number}

Key Information Identified:
{understanding}

Original Datasheet:
{datasheet_text}

Now create a JSON response with:
1. A detailed 120-150 word product description that sounds professional and highlights real features from the datasheet
2. All technical specifications you can extract with actual values and units

Return ONLY this JSON (no markdown, no explanations):
{{
  "seo_description": "Professional product description here highlighting real features, applications, and specifications from the datasheet. Make it compelling and detailed.",
  "technical_specifications": {{
    "Part Number": "{part_number}",
    "Manufacturer": "actual manufacturer name from datasheet",
    "Product Type": "specific product category",
    "Operating Voltage": "X.X V to Y.Y V" or specific value with unit,
    "Supply Current": "value with unit (mA, A, etc)",
    "Output Current": "value with unit if applicable",
    "Package Type": "exact package name (DIP, SOIC, QFN, etc)",
    "Pin Count": "number of pins",
    "Operating Temperature": "min to max with °C or °F",
    "Storage Temperature": "range if available",
    "Interface Type": "I2C, SPI, UART, USB, analog, etc",
    "Resolution": "if sensor/ADC/DAC",
    "Frequency": "operating frequency if applicable",
    "Memory": "RAM/ROM/Flash size if applicable",
    "Processor": "CPU type if microcontroller",
    "Dimensions": "physical size in mm",
    "Weight": "if available",
    "Key Features": [
      "Actual feature 1 from datasheet",
      "Actual feature 2 from datasheet", 
      "Actual feature 3 from datasheet"
    ],
    "Applications": [
      "Specific application 1 mentioned",
      "Specific application 2 mentioned"
    ],
    "Compliance": "certifications like CE, RoHS, FCC if mentioned"
  }}
}}

CRITICAL RULES:
- Use actual values from the datasheet, not generic descriptions
- Include units for all measurements (V, mA, MHz, mm, °C, etc)
- If a field is not in the datasheet, you can omit it entirely or use "N/A"
- The description MUST mention real specifications, not generic marketing
- Be specific with numbers and technical details"""

# -------------------------
# INTELLIGENT OLLAMA CALL WITH TWO-STAGE PROCESSING
# -------------------------
def call_ollama_intelligent(part_number, datasheet_text, max_retries=2):
    """Two-stage processing for better extraction"""
    
    if not datasheet_text or len(datasheet_text.strip()) < 100:
        datasheet_text = f"Limited datasheet information available for {part_number}. Extract what you can or use general knowledge for this component type."
    
    for attempt in range(max_retries):
        try:
            # STAGE 1: Understanding phase
            understanding_prompt = UNDERSTANDING_PROMPT.format(
                part_number=part_number,
                datasheet_text=datasheet_text[:3000]  # First chunk for understanding
            )
            
            understanding_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=understanding_prompt,
                keep_alive=0,
                options={
                    'num_ctx': 4096,
                    'temperature': 0.2,
                    'num_predict': 500,
                }
            )
            
            understanding = understanding_response['response'].strip()
            
            if DEBUG_MODE:
                print(f"\n🔍 {part_number} - Understanding: {understanding[:100]}...")
            
            # STAGE 2: Structured extraction with context
            extraction_prompt = EXTRACTION_PROMPT.format(
                part_number=part_number,
                understanding=understanding,
                datasheet_text=datasheet_text[:4000]  # More context for extraction
            )
            
            extraction_response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=extraction_prompt,
                format='json',
                keep_alive=0,
                options={
                    'num_ctx': 6144,  # Larger context for full extraction
                    'temperature': 0.1,
                    'num_predict': 2500,
                }
            )
            
            result_text = extraction_response['response'].strip()
            
            # Parse JSON with multiple fallback methods
            try:
                data = json.loads(result_text)
            except:
                # Method 1: Extract JSON with regex
                match = re.search(r'\{[\s\S]*\}', result_text)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except:
                        # Method 2: Try to fix common JSON issues
                        fixed = result_text.replace("'", '"').replace('\n', ' ')
                        match2 = re.search(r'\{[\s\S]*\}', fixed)
                        if match2:
                            data = json.loads(match2.group(0))
                        else:
                            raise ValueError("Could not extract valid JSON")
                else:
                    raise ValueError("No JSON structure found")
            
            # Validate we got actual data, not just N/A
            specs = data.get('technical_specifications', {})
            seo_desc = data.get('seo_description', '')
            
            # Check if extraction was successful (not all N/A)
            non_na_fields = sum(1 for v in specs.values() if v and str(v).upper() != "N/A" and v != [])
            
            if non_na_fields < 3 and len(seo_desc) < 50:
                if attempt < max_retries - 1:
                    print(f"\n⚠️  {part_number} - Low quality extraction, retrying...")
                    time.sleep(2)
                    continue
            
            if DEBUG_MODE:
                print(f"\n✓ {part_number} - Extracted {non_na_fields} specifications")
            
            gc.collect()
            
            specs_formatted = json.dumps(specs, indent=2, ensure_ascii=False)
            
            return {
                'seo_description': seo_desc if seo_desc else "Product information being updated",
                'technical_specifications': specs_formatted
            }
        
        except Exception as e:
            print(f"\n⚠️  Error for {part_number}: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying ({attempt+2}/{max_retries})...")
                time.sleep(3)
                continue
            return None
    
    return None

# -------------------------
# PROCESS SINGLE ROW
# -------------------------
def process_single_row(idx, row):
    """Process one product"""
    part_number = str(row.get('Product ID', f'ROW-{idx}'))
    datasheet_url = str(row.get('Datasheet URL', ''))
    
    try:
        # Extract text
        datasheet_text = extract_text_from_url(datasheet_url)
        
        if DEBUG_MODE and datasheet_text:
            print(f"\n📄 {part_number} - Extracted {len(datasheet_text.split())} words from datasheet")
        
        # Process with Ollama
        time.sleep(REQUEST_DELAY)
        result = call_ollama_intelligent(part_number, datasheet_text)
        
        if not result:
            return idx, None, None, "Extraction failed"
        
        return idx, result['seo_description'], result['technical_specifications'], None
    
    except Exception as e:
        return idx, None, None, str(e)

# -------------------------
# CHECK OLLAMA
# -------------------------
def check_ollama():
    """Verify Ollama setup"""
    try:
        # Try to list models
        response = ollama.list()
        
        # Handle different response formats
        if isinstance(response, dict):
            models_list = response.get('models', [])
        else:
            models_list = response
        
        # Extract model names safely
        model_names = []
        for m in models_list:
            if isinstance(m, dict):
                model_names.append(m.get('name', m.get('model', '')))
            else:
                model_names.append(str(m))
        
        # Check if our model exists
        model_exists = any(OLLAMA_MODEL in name for name in model_names)
        
        if not model_exists:
            print(f"❌ Model '{OLLAMA_MODEL}' not found!")
            print(f"\n📥 Download it now with:")
            print(f"   ollama pull {OLLAMA_MODEL}")
            print(f"\n💡 Available models: {', '.join(model_names) if model_names else 'None'}")
            
            # Try to pull the model automatically
            user_input = input(f"\n❓ Download {OLLAMA_MODEL} now? (y/n): ").lower()
            if user_input == 'y':
                print(f"\n⬇️  Downloading {OLLAMA_MODEL}...")
                import subprocess
                subprocess.run(['ollama', 'pull', OLLAMA_MODEL])
                print(f"✅ Download complete!")
                return True
            return False
        
        print(f"✅ Ollama ready with {OLLAMA_MODEL}")
        return True
    
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print(f"   Error details: {type(e).__name__}")
        
        # Try a simple test to see if Ollama is responsive
        try:
            test = ollama.generate(model=OLLAMA_MODEL, prompt="test", options={'num_predict': 1})
            if test:
                print("✅ Ollama is actually working! Continuing...")
                return True
        except:
            pass
        
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is running (check system tray)")
        print("   2. Try: ollama serve")
        print("   3. Verify model: ollama list")
        print(f"   4. Pull model: ollama pull {OLLAMA_MODEL}")
        return False

# -------------------------
# MAIN
# -------------------------
def main():
    print("=" * 70)
    print("🔬 ADVANCED ELECTRONICS DATASHEET EXTRACTION")
    print("=" * 70)
    print(f"Model: {OLLAMA_MODEL} (Two-Stage Intelligent Processing)")
    print(f"Strategy: Deep analysis → Structured extraction")
    print("=" * 70)
    
    if not check_ollama():
        return
    
    print(f"\n📂 Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   ✓ {len(df)} products loaded")
    
    if 'seo_description' not in df.columns:
        df['seo_description'] = ''
    if 'technical_specifications' not in df.columns:
        df['technical_specifications'] = ''
    
    # Find unprocessed rows
    rows_to_process = []
    for idx, row in df.iterrows():
        desc = str(df.at[idx, 'seo_description'])
        if len(desc) < 50 or desc == 'nan':
            rows_to_process.append((idx, row))
    
    total = len(rows_to_process)
    
    if total == 0:
        print("\n✅ All products already processed!")
        return
    
    print(f"\n🎯 Target: {total} products need processing")
    print(f"⏱️  Estimated time: ~{(total * 25 / 60 / MAX_WORKERS):.0f} minutes")
    print(f"💪 Using {MAX_WORKERS} parallel workers with intelligent extraction\n")
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_single_row, idx, row): idx
            for idx, row in rows_to_process
        }
        
        with tqdm(total=total, desc="Processing", unit="item", colour="green") as pbar:
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
                
                # Save frequently
                if (successful + failed) % SAVE_INTERVAL == 0:
                    df.to_csv(OUTPUT_PATH, index=False)
                    success_rate = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 0
                    tqdm.write(f"💾 Saved | Success rate: {success_rate:.0f}% ({successful}/{successful + failed})")
    
    # Final save
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Results
    success_rate = (successful / (successful + failed) * 100) if (successful + failed) > 0 else 0
    
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Successfully processed: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📊 Success rate: {success_rate:.1f}%")
    print(f"💰 Total cost: $0 (100% Free Local Processing)")
    print(f"📄 Output: {OUTPUT_PATH}")
    print("=" * 70)
    
    if success_rate < 80:
        print("\n💡 Tips to improve success rate:")
        print("   • Check if datasheets URLs are accessible")
        print("   • Verify Ollama has enough RAM (8GB+ recommended)")
        print("   • Try increasing REQUEST_DELAY to 1.5 seconds")

if __name__ == "__main__":
    main()