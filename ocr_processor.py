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
            # More lenient text validation - accept even small amounts
            if not ocr_text or len(ocr_text.strip()) < 5:
                print("⚠️ Insufficient text extracted from image")
                return None  # Return None to indicate invalid image
            
            # Check for receipt-like content - VERY lenient now
            receipt_keywords = ['total', 'price', 'item', 'qty', 'quantity', 'amount', 
                              'store', 'shop', 'receipt', 'purchased', 'date', '$', 
                              'card', 'cash', 'paid', 'rm', 'sgd', 'usd', 'eur', 'gbp',
                              'tax', 'subtotal', 'change', 'payment', 'bill', 'invoice', 
                              'merchant', 'tender', 'sale', 'transaction', 'purchase',
                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Numbers
                              '.', ',', ':', '-', 'x', '@', '#']  # Common receipt symbols
            text_lower = ocr_text.lower()
            
            # Count how many receipt keywords are present
            keyword_count = sum(1 for keyword in receipt_keywords if keyword in text_lower)
            
            # Very lenient - just check if there's ANY text that might be a receipt
            # Only reject if there are almost no keywords at all
            if keyword_count < 1:
                print(f"⚠️ No receipt-like content found")
                return None  # Definitely not a receipt
            
            # If we have at least 1 keyword, let GPT decide
            
            prompt = f"""You are an expert at identifying and parsing receipt data from OCR text.

Your task: Determine if this is a receipt/invoice/bill AND extract the data.

Receipt Text:
{ocr_text}

IMPORTANT INSTRUCTIONS:
1. Be VERY LENIENT in identifying receipts. Consider it a receipt if it has:
   - ANY mention of prices or amounts (even just numbers)
   - ANY store/business names or hints
   - ANY items or products listed
   - Even partial/blurry receipt information

2. ONLY reject if it's clearly NOT a receipt, such as:
   - Personal letters or emails
   - Book pages or articles
   - Random street signs
   - Photos with no transaction information at all

3. If you're unsure, ASSUME it's a receipt and try to extract data.

If this is clearly NOT a receipt, return:
{{"is_receipt": false, "reason": "brief explanation"}}

If this IS or MIGHT BE a receipt (even if very unclear), extract whatever you can:
{{
    "is_receipt": true,
    "shop_name": "store name or 'Unknown Store'",
    "date": "YYYY-MM-DD or today's date",
    "payment_mode": "Cash/Credit Card/Debit Card/E-Wallet/Other",
    "items": [
        {{"name": "item name or generic 'Item 1'", "quantity": 1, "unit_price": 10.50, "total_price": 10.50}}
    ]
}}

EXTRACTION RULES:
- Extract ALL items you can find
- If only prices visible (no item names), create generic items: "Item 1", "Item 2", etc.
- If only total visible, create one item with that total
- Quantities can have 3 decimals (e.g., 2.500 kg)
- Prices can be negative (discounts)
- Return ONLY JSON, no markdown or explanations
- When in doubt, TRY TO EXTRACT rather than reject
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a lenient receipt validator. Assume images are receipts unless clearly not (like letters, books, signs). For receipts, extract any data you can find, even if partial or unclear. When in doubt, try to extract rather than reject."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Higher for more flexible interpretation
                max_tokens=1500
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
            
            # Check if GPT determined this is NOT a receipt
            if not data.get('is_receipt', True):
                reason = data.get('reason', 'Not a valid receipt image')
                print(f"❌ Not a receipt: {reason}")
                return None  # Return None to indicate invalid image
            
            # Validate extraction quality - be more lenient for actual receipts
            shop_name = str(data.get('shop_name', 'Unknown Store')) or 'Unknown Store'
            items = data.get('items', [])
            
            # Only reject if COMPLETELY empty or all zeros with "Unknown" labels
            if len(items) == 0:
                print("⚠️ No items extracted but appears to be a receipt")
                # Still might be a receipt, just very blurry - allow manual entry
                return [self._get_default_data()]
            
            # Accept if we have ANY non-zero prices OR any non-"Unknown" item names
            has_useful_data = any(
                item.get('total_price', 0) != 0 or 
                item.get('unit_price', 0) != 0 or
                (item.get('name', 'Unknown Item') != 'Unknown Item' and item.get('name', '') != '')
                for item in items
            )
            
            if not has_useful_data and shop_name == 'Unknown Store':
                print("⚠️ Could not extract any useful data from receipt")
                return [self._get_default_data()]
            
            # If we got here, we have at least SOME data - keep it!
            
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
            print(f"Response was: {result[:200] if 'result' in locals() else 'N/A'}")
            return None  # Invalid response
            
        except Exception as e:
            print(f"❌ GPT parsing error: {e}")
            return None  # Error occurred
    
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
        VERY lenient version - accepts most images that might be receipts
        Only rejects obvious non-receipts (letters, books, signs)
        
        Returns: List of records (one per item), or None if clearly not a receipt
        """
        print(f"🔍 Processing with Tesseract OCR: {image_path}")
        
        # Step 1: Extract text using Tesseract (FREE!)
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text or len(ocr_text.strip()) < 3:
            print("⚠️ OCR failed or almost no text extracted")
            # Even with minimal text, try to process - might be very faded receipt
            # Return default data for manual entry
            return [self._get_default_data()]
        
        print(f"✅ Tesseract extracted {len(ocr_text)} characters")
        print(f"📄 Preview: {ocr_text[:200]}...")
        
        # Step 2: Parse text with GPT-4o-mini (very lenient validation + extraction)
        records = self.parse_receipt_with_gpt(ocr_text)
        
        # None means GPT determined it's clearly NOT a receipt
        if records is None:
            return None
        
        return records
