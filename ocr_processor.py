import os
import re
import json
import base64
from datetime import datetime
from PIL import Image
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OCRProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._setup_tesseract()
    
    def _setup_tesseract(self):
        """Setup Tesseract path for different systems"""
        tesseract_paths = [
            '/usr/local/bin/tesseract',
            '/usr/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        ]
        
        env_path = os.getenv('TESSERACT_PATH')
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            return
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return
    
    def extract_text_from_image(self, image_path):
        """Extract text from receipt image using Tesseract OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract not found")
            return None
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def parse_receipt_with_gpt(self, ocr_text):
        """Use GPT-4 to extract ALL items from receipt"""
        try:
            prompt = f"""You are an expert at parsing receipt data. Extract ALL items from this receipt.

Receipt Text:
{ocr_text}

Return ONLY valid JSON with this structure:
{{
    "shop_name": "store name",
    "date": "YYYY-MM-DD",
    "payment_mode": "Cash/Credit Card/Debit Card/E-Wallet/Other",
    "items": [
        {{"name": "item 1", "quantity": 1, "unit_price": 10.50, "total_price": 10.50}},
        {{"name": "item 2", "quantity": 2, "unit_price": 5.00, "total_price": 10.00}}
    ]
}}

CRITICAL RULES:
1. Extract ALL items, not just the first one
2. Each item needs: name, quantity, unit_price, total_price
3. Return ONLY JSON, no markdown, no code blocks
4. All prices are numbers without $ symbols
5. Date in YYYY-MM-DD format
6. If info missing, use defaults: shop_name="Unknown Store", date=today, payment_mode="Other"
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a receipt parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean markdown
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            result = result.strip()
            print(f"GPT Response: {result[:200]}...")
            
            data = json.loads(result)
            
            # Parse into multiple records (one per item)
            records = []
            shop_name = str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store'
            date = self._validate_date(data.get('date'))
            payment_mode = self._validate_mode(data.get('payment_mode', 'Other'))
            
            items = data.get('items', [])
            
            if not items:
                # Fallback: create single item
                return [self._get_default_data()]
            
            for item in items:
                record = {
                    'shop_name': shop_name,
                    'date': date,
                    'item': str(item.get('name', 'Unknown Item')) or 'Unknown Item',
                    'mode': payment_mode,
                    'unit': max(1, int(float(item.get('quantity', 1) or 1))),
                    'unit_price': abs(float(item.get('unit_price', 0) or 0)),
                    'total_price': abs(float(item.get('total_price', 0) or 0))
                }
                
                # Calculate missing prices
                if record['unit_price'] == 0 and record['total_price'] > 0:
                    record['unit_price'] = record['total_price'] / record['unit']
                
                if record['total_price'] == 0 and record['unit_price'] > 0:
                    record['total_price'] = record['unit_price'] * record['unit']
                
                records.append(record)
            
            print(f"✅ Extracted {len(records)} items from {shop_name}")
            return records
            
        except Exception as e:
            print(f"GPT Parsing Error: {e}")
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
        
        for valid_mode in valid_modes:
            if mode_str.lower() == valid_mode.lower():
                return valid_mode
        
        mode_lower = mode_str.lower()
        if 'cash' in mode_lower:
            return 'Cash'
        elif 'credit' in mode_lower or 'visa' in mode_lower or 'mastercard' in mode_lower:
            return 'Credit Card'
        elif 'debit' in mode_lower:
            return 'Debit Card'
        elif 'wallet' in mode_lower or 'paypal' in mode_lower or 'apple pay' in mode_lower:
            return 'E-Wallet'
        
        return 'Other'
    
    def _get_default_data(self):
        """Return default data structure"""
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
        """Main processing pipeline - returns LIST of records"""
        print(f"Processing: {image_path}")
        
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text or len(ocr_text.strip()) < 10:
            print("⚠️ OCR failed or insufficient text")
            return [self._get_default_data()]
        
        print(f"✅ OCR extracted {len(ocr_text)} characters")
        
        # Returns list of records (one per item)
        records = self.parse_receipt_with_gpt(ocr_text)
        return records


class OCRProcessorVision:
    """Vision API - More accurate, extracts ALL items"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def process_receipt(self, image_path):
        """Process receipt using GPT-4 Vision - returns LIST of records"""
        print(f"Processing with Vision API: {image_path}")
        
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = """Analyze this receipt image and extract ALL items purchased.

Return ONLY valid JSON in this exact format:
{
    "shop_name": "store name",
    "date": "YYYY-MM-DD",
    "payment_mode": "Cash/Credit Card/Debit Card/E-Wallet/Other",
    "items": [
        {"name": "item 1 name", "quantity": 1, "unit_price": 10.50, "total_price": 10.50},
        {"name": "item 2 name", "quantity": 2, "unit_price": 5.00, "total_price": 10.00}
    ]
}

CRITICAL RULES:
1. Extract ALL items from the receipt, not just one
2. Each item must have: name, quantity, unit_price, total_price
3. Return ONLY the JSON object - no markdown, no code blocks, no ```json
4. All prices are numbers without $ or currency symbols
5. Date in YYYY-MM-DD format
6. Payment mode must be exactly one of: Cash, Credit Card, Debit Card, E-Wallet, Other
7. If multiple items with same name, keep them separate with quantities
8. Calculate unit_price = total_price / quantity if needed

Example for 3 items:
{
  "shop_name": "Walmart",
  "date": "2024-01-15",
  "payment_mode": "Credit Card",
  "items": [
    {"name": "Apple", "quantity": 3, "unit_price": 1.50, "total_price": 4.50},
    {"name": "Bread", "quantity": 1, "unit_price": 3.99, "total_price": 3.99},
    {"name": "Milk", "quantity": 2, "unit_price": 4.50, "total_price": 9.00}
  ]
}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean markdown
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            result = result.strip()
            print(f"Vision API Response: {result[:300]}...")
            
            data = json.loads(result)
            
            # Parse into multiple records (one per item)
            records = []
            shop_name = str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store'
            date = self._validate_date(data.get('date'))
            payment_mode = self._validate_mode(data.get('payment_mode', 'Other'))
            
            items = data.get('items', [])
            
            if not items:
                print("⚠️ No items found, creating default")
                return [self._get_default_data()]
            
            for item in items:
                item_name = str(item.get('name', 'Unknown Item')) or 'Unknown Item'
                quantity = max(1, int(float(item.get('quantity', 1) or 1)))
                unit_price = abs(float(item.get('unit_price', 0) or 0))
                total_price = abs(float(item.get('total_price', 0) or 0))
                
                # Calculate missing prices
                if unit_price == 0 and total_price > 0:
                    unit_price = total_price / quantity
                
                if total_price == 0 and unit_price > 0:
                    total_price = unit_price * quantity
                
                record = {
                    'shop_name': shop_name,
                    'date': date,
                    'item': item_name,
                    'mode': payment_mode,
                    'unit': quantity,
                    'unit_price': unit_price,
                    'total_price': total_price
                }
                
                records.append(record)
                print(f"  → {item_name} x{quantity} = ${total_price:.2f}")
            
            print(f"✅ Extracted {len(records)} items from {shop_name}")
            return records
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Response was: {result[:500]}")
            return [self._get_default_data()]
        except Exception as e:
            print(f"❌ Vision API Error: {e}")
            import traceback
            traceback.print_exc()
            return [self._get_default_data()]
    
    def _validate_date(self, date_str):
        """Validate and parse date string"""
        if not date_str:
            return datetime.now()
        
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
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
        
        for valid_mode in valid_modes:
            if mode_str.lower() == valid_mode.lower():
                return valid_mode
        
        mode_lower = mode_str.lower()
        if 'cash' in mode_lower:
            return 'Cash'
        elif 'credit' in mode_lower or 'visa' in mode_lower or 'mastercard' in mode_lower:
            return 'Credit Card'
        elif 'debit' in mode_lower:
            return 'Debit Card'
        elif 'wallet' in mode_lower or 'paypal' in mode_lower or 'apple pay' in mode_lower:
            return 'E-Wallet'
        
        return 'Other'
    
    def _get_default_data(self):
        """Return default data structure"""
        return {
            'shop_name': 'Unknown Store',
            'date': datetime.now(),
            'item': 'Unknown Item',
            'mode': 'Other',
            'unit': 1,
            'unit_price': 0.0,
            'total_price': 0.0
        }
