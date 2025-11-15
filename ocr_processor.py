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
        # Try common Tesseract paths
        tesseract_paths = [
            '/usr/local/bin/tesseract',  # macOS Homebrew
            '/usr/bin/tesseract',         # Linux
            '/opt/homebrew/bin/tesseract', # macOS ARM
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  # Windows
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        ]
        
        # Check environment variable first
        env_path = os.getenv('TESSERACT_PATH')
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            return
        
        # Try each path
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
            print("ERROR: Tesseract not found. Please install it:")
            print("  macOS: brew install tesseract")
            print("  Ubuntu: sudo apt-get install tesseract-ocr")
            print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            return None
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def parse_receipt_with_gpt(self, ocr_text):
        """Use GPT-4 to extract structured data from OCR text"""
        try:
            prompt = f"""You are an expert at parsing receipt data. Extract the following information from this receipt text and return ONLY valid JSON.

Receipt Text:
{ocr_text}

Extract these fields:
- shop_name: The store/shop name (string)
- date: Date in YYYY-MM-DD format (string)
- item: The main item or first item purchased (string)
- mode: Payment method - must be one of: "Cash", "Credit Card", "Debit Card", "E-Wallet", "Other" (string)
- unit: Number of units purchased (integer, default 1)
- unit_price: Price per unit in dollars (float, no currency symbol)
- total_price: Total amount in dollars (float, no currency symbol)

IMPORTANT RULES:
1. Return ONLY valid JSON, no other text, no markdown, no code blocks
2. All prices must be numbers without $ or currency symbols
3. Date must be YYYY-MM-DD format
4. Mode must be exactly one of the five options above
5. If you can't find a value, use these defaults:
   - shop_name: "Unknown Store"
   - date: today's date
   - item: "Unknown Item"
   - mode: "Other"
   - unit: 1
   - unit_price: 0.0
   - total_price: 0.0

Return format:
{{"shop_name": "Store Name", "date": "2024-01-15", "item": "Product Name", "mode": "Credit Card", "unit": 1, "unit_price": 10.50, "total_price": 10.50}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a receipt data extraction expert. Return only valid JSON with no additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up response - remove markdown if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing whitespace or newlines
            result = result.strip()
            
            print(f"GPT Response: {result}")
            
            # Parse JSON
            data = json.loads(result)
            
            # Validate and clean data with strict type checking
            cleaned_data = {
                'shop_name': str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store',
                'date': self._validate_date(data.get('date')),
                'item': str(data.get('item', 'Unknown Item')) or 'Unknown Item',
                'mode': self._validate_mode(data.get('mode', 'Other')),
                'unit': max(1, int(float(data.get('unit', 1) or 1))),
                'unit_price': abs(float(data.get('unit_price', 0) or 0)),
                'total_price': abs(float(data.get('total_price', 0) or 0))
            }
            
            # If unit_price is 0 but total_price exists, calculate unit_price
            if cleaned_data['unit_price'] == 0 and cleaned_data['total_price'] > 0:
                cleaned_data['unit_price'] = cleaned_data['total_price'] / cleaned_data['unit']
            
            # If total_price is 0 but unit_price exists, calculate total_price
            if cleaned_data['total_price'] == 0 and cleaned_data['unit_price'] > 0:
                cleaned_data['total_price'] = cleaned_data['unit_price'] * cleaned_data['unit']
            
            print(f"✅ Extracted: {cleaned_data['shop_name']} - {cleaned_data['item']} - ${cleaned_data['total_price']:.2f}")
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response was: {result}")
            return self._get_default_data()
        except Exception as e:
            print(f"GPT Parsing Error: {e}")
            return self._get_default_data()
    
    def _validate_date(self, date_str):
        """Validate and parse date string"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try parsing as YYYY-MM-DD
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                # Try other common formats
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
        
        # Check for exact match (case insensitive)
        for valid_mode in valid_modes:
            if mode_str.lower() == valid_mode.lower():
                return valid_mode
        
        # Check for partial matches
        mode_lower = mode_str.lower()
        if 'cash' in mode_lower:
            return 'Cash'
        elif 'credit' in mode_lower or 'visa' in mode_lower or 'mastercard' in mode_lower or 'amex' in mode_lower:
            return 'Credit Card'
        elif 'debit' in mode_lower:
            return 'Debit Card'
        elif 'wallet' in mode_lower or 'paypal' in mode_lower or 'venmo' in mode_lower or 'apple pay' in mode_lower or 'google pay' in mode_lower:
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
        """Main processing pipeline"""
        print(f"Processing: {image_path}")
        
        # Step 1: OCR
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text or len(ocr_text.strip()) < 10:
            print("⚠️ OCR failed or insufficient text extracted, using defaults")
            return self._get_default_data()
        
        print(f"✅ OCR extracted {len(ocr_text)} characters")
        
        # Step 2: Parse with GPT
        structured_data = self.parse_receipt_with_gpt(ocr_text)
        
        return structured_data


# Vision API method - more accurate, recommended
class OCRProcessorVision:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def process_receipt(self, image_path):
        """Process receipt using GPT-4 Vision API"""
        print(f"Processing with Vision API: {image_path}")
        
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = """Analyze this receipt image and extract the following information.

You must return ONLY valid JSON with these exact fields:
{
    "shop_name": "name of store/shop",
    "date": "YYYY-MM-DD format",
    "item": "main item or first item listed",
    "mode": "must be exactly one of: Cash, Credit Card, Debit Card, E-Wallet, or Other",
    "unit": integer number of units,
    "unit_price": float price per unit (no $ symbol),
    "total_price": float total amount (no $ symbol)
}

CRITICAL RULES:
1. Return ONLY the JSON object, no other text
2. No markdown formatting, no code blocks, no ```json
3. All prices must be numbers without currency symbols
4. Date must be YYYY-MM-DD format
5. Mode must be exactly: "Cash", "Credit Card", "Debit Card", "E-Wallet", or "Other"
6. If you see multiple items, pick the main/first item
7. If info is unclear, use these defaults:
   - shop_name: "Unknown Store"
   - date: "2024-01-01"
   - item: "Purchase"
   - mode: "Other"
   - unit: 1
   - unit_price: 0.0
   - total_price: 0.0

Example valid response:
{"shop_name": "Walmart", "date": "2024-01-15", "item": "Office Supplies", "mode": "Credit Card", "unit": 1, "unit_price": 45.99, "total_price": 45.99}
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
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing whitespace
            result = result.strip()
            
            print(f"Vision API Response: {result}")
            
            # Parse JSON
            data = json.loads(result)
            
            # Clean and validate with strict types
            cleaned_data = {
                'shop_name': str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store',
                'date': self._validate_date(data.get('date', '2024-01-01')),
                'item': str(data.get('item', 'Purchase')) or 'Purchase',
                'mode': self._validate_mode(data.get('mode', 'Other')),
                'unit': max(1, int(float(data.get('unit', 1) or 1))),
                'unit_price': abs(float(data.get('unit_price', 0) or 0)),
                'total_price': abs(float(data.get('total_price', 0) or 0))
            }
            
            # Calculate missing prices
            if cleaned_data['unit_price'] == 0 and cleaned_data['total_price'] > 0:
                cleaned_data['unit_price'] = cleaned_data['total_price'] / cleaned_data['unit']
            
            if cleaned_data['total_price'] == 0 and cleaned_data['unit_price'] > 0:
                cleaned_data['total_price'] = cleaned_data['unit_price'] * cleaned_data['unit']
            
            print(f"✅ Extracted: {cleaned_data['shop_name']} - {cleaned_data['item']} - ${cleaned_data['total_price']:.2f}")
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Response was: {result}")
            return self._get_default_data()
        except Exception as e:
            print(f"❌ Vision API Error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_data()
    
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
        
        # Check for exact match
        for valid_mode in valid_modes:
            if mode_str.lower() == valid_mode.lower():
                return valid_mode
        
        # Check for partial matches
        mode_lower = mode_str.lower()
        if 'cash' in mode_lower:
            return 'Cash'
        elif 'credit' in mode_lower or 'visa' in mode_lower or 'mastercard' in mode_lower or 'amex' in mode_lower:
            return 'Credit Card'
        elif 'debit' in mode_lower:
            return 'Debit Card'
        elif 'wallet' in mode_lower or 'paypal' in mode_lower or 'venmo' in mode_lower or 'apple pay' in mode_lower or 'google pay' in mode_lower:
            return 'E-Wallet'
        
        return 'Other'
    
    def _get_default_data(self):
        """Return default data structure"""
        return {
            'shop_name': 'Unknown Store',
            'date': datetime.now(),
            'item': 'Purchase',
            'mode': 'Other',
            'unit': 1,
            'unit_price': 0.0,
            'total_price': 0.0
        }