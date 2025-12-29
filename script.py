import pandas as pd
import requests
import pdfplumber
from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
import os

# -------------------------
# CONFIGURATION
# -------------------------
CSV_PATH = "Master_Inventory - Master_Inventory.csv"
OUTPUT_PATH = "Master_Inventory_enriched.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "ENTER_YOUR_GROQ_API_KEY")

# Model settings - BEST OPTIONS:
# "llama-3.3-70b-versatile" - Most accurate, best for technical content (RECOMMENDED)
# "llama-3.1-70b-versatile" - Alternative if 3.3 has issues
# "mixtral-8x7b-32768" - Faster but slightly less accurate
MODEL_NAME = "llama-3.3-70b-versatile"

REQUEST_DELAY = 20  # Groq is VERY fast with high rate limits
MAX_WORKERS = 2  # High parallel processing
SAVE_INTERVAL = 5  # Save progress every N rows

# Configure Groq
client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# TEXT EXTRACTION
# -------------------------
def extract_text_from_url(url):
    """Extract text from PDF or HTML datasheet"""
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
                # Groq handles up to 32k tokens well
                words = full_text.split()[:4000]
                return " ".join(words)
        
        # Handle HTML/web pages
        else:
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            words = text.split()[:4000]
            return " ".join(words)
    
    except Exception as e:
        print(f"\n⚠️  Error extracting from {url[:50]}... - {str(e)}")
        return None

# -------------------------
# PROMPT
# -------------------------
EXTRACTION_PROMPT = """You are an electronics technical documentation expert. Analyze this datasheet and extract information.

Product Information:
- Part Number: {part_number}
- Product Name: {product_name}
- Category: {category}

Datasheet Content:
{datasheet_text}

Return ONLY valid JSON with this structure (no markdown, no code blocks, no extra text):

{{
  "seo_description": "Write a compelling 120-150 word product description for e-commerce. Include key features, applications, benefits in professional but accessible language. No line breaks in the description.",
  "technical_specifications": {{
    "Part Number": "{part_number}",
    "Product Type": "value or N/A",
    "Manufacturer": "value or N/A",
    "Operating Voltage": "value or N/A",
    "Current Rating": "value or N/A",
    "Power Rating": "value or N/A",
    "Package Type": "value or N/A",
    "Temperature Range": "value or N/A",
    "Interface": "value or N/A",
    "Key Features": ["feature1", "feature2"],
    "Applications": ["app1", "app2"],
    "Dimensions": "value or N/A"
  }}
}}

CRITICAL RULES:
1. Extract ONLY explicitly stated information from the datasheet
2. Use "N/A" for missing specifications
3. Return ONLY the JSON object - no explanations, no markdown formatting
4. Ensure all JSON strings are properly escaped
5. Keep seo_description as a single paragraph without line breaks"""

# -------------------------
# GROQ API CALL
# -------------------------
def call_groq(part_number, product_name, category, datasheet_text):
    """Call Groq API to extract specs and generate description"""
    try:
        prompt = EXTRACTION_PROMPT.format(
            part_number=part_number,
            product_name=product_name,
            category=category,
            datasheet_text=datasheet_text
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical documentation expert. Always return valid JSON only, with no markdown formatting or code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2500,
            response_format={"type": "json_object"}  # Force JSON mode
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Aggressive JSON cleaning (just in case)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        # Find actual JSON boundaries
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            result_text = result_text[start_idx:end_idx+1]
        
        result_text = result_text.strip()
        
        # Parse JSON
        data = json.loads(result_text)
        
        return {
            'seo_description': data.get('seo_description', 'N/A'),
            'technical_specifications': json.dumps(
                data.get('technical_specifications', {}), 
                indent=2
            )
        }
    
    except json.JSONDecodeError as e:
        print(f"\n⚠️  JSON parse error for {part_number}: {e}")
        print(f"   Response preview: {result_text[:200] if 'result_text' in locals() else 'N/A'}...")
        return None
    except Exception as e:
        print(f"\n⚠️  API error for {part_number}: {e}")
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
        # Extract datasheet text
        datasheet_text = extract_text_from_url(datasheet_url)
        
        if not datasheet_text or len(datasheet_text.strip()) < 200:
            return idx, None, None, f"Insufficient text extracted ({len(datasheet_text) if datasheet_text else 0} chars)"
        
        # Call Groq
        time.sleep(REQUEST_DELAY)
        result = call_groq(part_number, product_name, category, datasheet_text)
        
        if not result:
            return idx, None, None, "API returned no data"
        
        return idx, result['seo_description'], result['technical_specifications'], None
    
    except Exception as e:
        return idx, None, None, str(e)

# -------------------------
# MAIN PROCESSING
# -------------------------
def main():
    print("=" * 60)
    print("Electronic Components Datasheet Enrichment")
    print(f"Provider: Groq (FREE)")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    
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
    
    # Estimate time
    estimated_minutes = (total * REQUEST_DELAY) / 60 / MAX_WORKERS
    print(f"⏱️  Estimated time: ~{estimated_minutes:.1f} minutes")
    
    # Process rows
    successful = 0
    failed = 0
    
    print(f"\n⚙️  Processing with {MAX_WORKERS} parallel workers...\n")
    
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
    print(f"⚡ Model: {MODEL_NAME}")
    print(f"💰 Cost: FREE")
    print(f"📄 Output: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()