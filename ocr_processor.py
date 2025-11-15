import os
import json
from datetime import datetime
from PIL import Image
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OCRProcessor:
    """Tesseract-based OCR Processor - Completely FREE!"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """Setup Tesseract path for different systems"""
        tesseract_paths = [
            '/usr/local/bin/tesseract',  # macOS Intel
            '/usr/bin/tesseract',         # Linux
            '/opt/homebrew/bin/tesseract', # macOS ARM (M1/M2/M3)
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  # Windows 64-bit
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',  # Windows 32-bit
        ]
        
        # Check environment variable first
        env_path = os.getenv('TESSERACT_PATH')
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            print(f"✅ Using Tesseract from: {env_path}")
            return
        
        # Try common paths
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✅ Using Tesseract from: {path}")
                return
        
        print("⚠️ Tesseract not found in common locations. Set TESSERACT_PATH in .env")
    
    def extract_text_from_image(self, image_path):
        """Extract text from receipt image using Tesseract OCR (FREE!)"""
        try:
            image = Image.open(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            return text
            
        except pytesseract.TesseractNotFoundError:
            print("❌ ERROR: Tesseract not found!")
            print("Please install Tesseract:")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  macOS: brew install tesseract")
            print("  Linux: sudo apt-get install tesseract-ocr")
            return None
            
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            return None
    
    def parse_receipt_with_gpt(self, ocr_text):
        """Use GPT-4o-mini to parse OCR text into structured data"""
        try:
            # Validate text quality
            if not ocr_text or len(ocr_text.strip()) < 20:
                print("⚠️ Insufficient text extracted from image")
                return [self._get_default_data()]
            
            # Check if text looks like a receipt
            receipt_keywords = ['total', 'price', 'item', 'qty', 'quantity', 'amount', 
                              'store', 'shop', 'receipt', 'purchased', 'date', '$', 
                              'card', 'cash', 'paid']
            text_lower = ocr_text.lower()
            has_receipt_content = any(keyword in text_lower for keyword in receipt_keywords)
            
            if not has_receipt_content:
                print("⚠️ Text does not appear to be a receipt")
                return [self._get_default_data()]
            
            prompt = f"""You are an expert at parsing receipt data. Extract ALL items from this receipt text.

Receipt Text:
{ocr_text}

CRITICAL INSTRUCTIONS:
- If you CANNOT confidently extract the data, return ONLY this JSON:
  {{"shop_name": "Unknown Store", "date": "2024-01-01", "payment_mode": "Other", "items": [{{"name": "Unknown Item", "quantity": 1, "unit_price": 0.0, "total_price": 0.0}}]}}

- If you CAN extract data, return valid JSON in this format:
{{
    "shop_name": "store name from receipt",
    "date": "YYYY-MM-DD",
    "payment_mode": "Cash/Credit Card/Debit Card/E-Wallet/Other",
    "items": [
        {{"name": "item name", "quantity": 1.5, "unit_price": 10.50, "total_price": 15.75}},
        {{"name": "another item", "quantity": 2, "unit_price": 5.00, "total_price": 10.00}}
    ]
}}

RULES:
1. Extract ALL items from the receipt, not just one
2. Return ONLY actual data you can see - DO NOT guess or make up information
3. If any field is unclear, use defaults (Unknown Store, Other, 0.0)
4. Return ONLY JSON - no markdown, no code blocks, no explanation
5. Quantities can have up to 3 decimal places (e.g., 2.500 kg)
6. Unit prices can be negative for discounts (e.g., -5.00)
7. For each item, ensure: unit_price × quantity = total_price
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a receipt data extraction expert. Return only valid JSON. Never make up data - if unsure, use Unknown/Other defaults."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean markdown formatting if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            result = result.strip()
            print(f"📝 GPT parsing result: {result[:150]}...")
            
            # Parse JSON
            data = json.loads(result)
            
            # Validate extraction quality - reject if we got defaults
            shop_name = str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store'
            items = data.get('items', [])
            
            if (shop_name == 'Unknown Store' and 
                len(items) == 1 and 
                items[0].get('name') == 'Unknown Item' and
                items[0].get('total_price') == 0.0):
                print("⚠️ Could not extract valid receipt data")
                return [self._get_default_data()]
            
            # Convert to record format
            records = []
            date = self._validate_date(data.get('date'))
            payment_mode = self._validate_mode(data.get('payment_mode', 'Other'))
            
            if not items:
                return [self._get_default_data()]
            
            for item in items:
                quantity = max(0.001, float(item.get('quantity', 1) or 1))
                unit_price = float(item.get('unit_price', 0) or 0)  # Can be negative
                total_price = float(item.get('total_price', 0) or 0)
                
                # Calculate missing prices
                if unit_price == 0 and total_price != 0:
                    unit_price = total_price / quantity
                elif total_price == 0 and unit_price != 0:
                    total_price = unit_price * quantity
                elif unit_price != 0 and total_price != 0:
                    expected_total = unit_price * quantity
                    if abs(expected_total - total_price) > 0.01:
                        total_price = expected_total
                
                record = {
                    'shop_name': shop_name,
                    'date': date,
                    'item': str(item.get('name', 'Unknown Item')) or 'Unknown Item',
                    'mode': payment_mode,
                    'unit': round(quantity, 3),  # 3 decimal places for weights
                    'unit_price': round(unit_price, 2),
                    'total_price': round(total_price, 2)
                }
                
                records.append(record)
            
            print(f"✅ Extracted {len(records)} items from {shop_name}")
            return records
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response was: {result[:200]}")
            return [self._get_default_data()]
            
        except Exception as e:
            print(f"❌ GPT parsing error: {e}")
            return [self._get_default_data()]
    
    def _validate_date(self, date_str):
        """Validate and parse date string"""
        if not date_str:
            return datetime.now()
        
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d', '%m-%d-%Y']:
                    try:
                        return datetime.strptime(str(date_str), fmt)
                    except:
                        continue
            except:
                pass
        
        return datetime.now()
    
    def _validate_mode(self, mode_str):
        """Validate payment mode"""
        valid_modes = ['Cash', 'Credit Card', 'Debit Card', 'E-Wallet', 'Other']
        
        if not mode_str:
            return 'Other'
        
        mode_str = str(mode_str).strip()
        
        # Check exact match
        for valid_mode in valid_modes:
            if mode_str.lower() == valid_mode.lower():
                return valid_mode
        
        # Check partial matches
        mode_lower = mode_str.lower()
        if 'cash' in mode_lower:
            return 'Cash'
        elif 'credit' in mode_lower or 'visa' in mode_lower or 'mastercard' in mode_lower or 'amex' in mode_lower:
            return 'Credit Card'
        elif 'debit' in mode_lower:
            return 'Debit Card'
        elif 'wallet' in mode_lower or 'paypal' in mode_lower or 'apple pay' in mode_lower or 'google pay' in mode_lower:
            return 'E-Wallet'
        
        return 'Other'
    
    def _get_default_data(self):
        """Return default data structure when extraction fails"""
        return {
            'shop_name': 'Unknown Store',
            'date': datetime.now(),
            'item': 'Unknown Item',
            'mode': 'Other',
            'unit': 1,
            'unit_price': 0.0,
            'total_price': 0.0
        }
    
    def process_receipt(self, image_path):
        """
        Main processing pipeline using Tesseract OCR (FREE!)
        
        Returns: List of records (one per item)
        """
        print(f"🔍 Processing with Tesseract OCR: {image_path}")
        
        # Step 1: Extract text using Tesseract (FREE!)
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text or len(ocr_text.strip()) < 10:
            print("⚠️ OCR failed or insufficient text extracted")
            return [self._get_default_data()]
        
        print(f"✅ Tesseract extracted {len(ocr_text)} characters")
        
        # Step 2: Parse text with GPT-4o-mini
        records = self.parse_receipt_with_gpt(ocr_text)
        
        return records
