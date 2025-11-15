# 📋 Receipt Verification System with RAG Chatbot

A comprehensive AI-powered receipt management system for businesses to track, verify, and analyze staff purchases. Features OCR scanning, intelligent chatbot queries, and advanced analytics dashboards.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🔍 **Smart OCR Processing**
- Automatic text extraction from receipt images using Tesseract OCR
- AI-powered data parsing with GPT-4 for structured extraction
- Support for multiple receipt formats and layouts
- Alternative Vision API integration for higher accuracy

### 💬 **RAG-Powered Chatbot**
- Natural language queries on receipt database
- Semantic search using OpenAI embeddings
- Context-aware responses with relevant receipt retrieval
- ChromaDB vector storage for efficient similarity search

### 📊 **Advanced Analytics**
- Real-time spending trends and visualizations
- Automatic purchase categorization
- Multi-dimensional analysis (by shop, date, category, payment mode)
- Interactive Plotly charts and graphs
- Exportable CSV reports for management

### 🎯 **User-Friendly Interface**
- Clean, intuitive Streamlit web interface
- Three main sections: Upload, Query, Analytics
- Mobile-responsive design
- Real-time data verification and editing

## 🎬 Demo

### Upload & Verify Receipts
```
1. Upload receipt image → 2. AI extracts data → 3. Verify & save → 4. Instant analysis
```

### Query with Natural Language
```
User: "Show me all Walmart purchases from last month"
Bot: "I found 5 receipts from Walmart in January totaling $234.56..."
[Displays relevant receipts table]
```

### View Analytics Dashboard
```
📊 Total Spending: $1,234.56 | 🧾 Receipts: 45 | 🏪 Shops: 12
[Interactive charts showing trends, categories, payment modes]
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/receipt-verification.git
cd receipt-verification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Configure environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Create data directories
mkdir -p data/uploads

# Run tests
python test_system.py

# Start application
streamlit run app.py
```

Visit `http://localhost:8501` to use the application.

## 📖 Documentation

### Project Structure
```
receipt-verification/
├── app.py                 # Main Streamlit application
├── ocr_processor.py       # OCR and GPT extraction logic
├── rag_chatbot.py         # RAG implementation with ChromaDB
├── analytics.py           # Analytics and visualization module
├── test_system.py         # Comprehensive test suite
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
└── data/
    ├── receipts.csv      # Structured receipt data (auto-created)
    ├── uploads/          # Uploaded receipt images
    └── chroma_db/       # Vector database storage
```

### Core Components

#### 1. OCR Processor (`ocr_processor.py`)
```python
class OCRProcessor:
    - extract_text_from_image()  # Tesseract OCR
    - parse_receipt_with_gpt()   # GPT-4 parsing
    - process_receipt()          # Main pipeline
```

**Features:**
- Tesseract OCR for text extraction
- GPT-4 for intelligent data structuring
- Automatic date parsing and validation
- Fallback to Vision API (optional)

#### 2. RAG Chatbot (`rag_chatbot.py`)
```python
class RAGChatbot:
    - add_receipt()       # Add to vector DB
    - query()            # Semantic search + GPT
    - rebuild_index()    # Rebuild from CSV
```

**Features:**
- OpenAI embeddings (text-embedding-3-small)
- ChromaDB for vector storage
- Context-aware GPT-4 responses
- Relevant receipt retrieval

#### 3. Analytics (`analytics.py`)
```python
class ReceiptAnalytics:
    - categorize_purchases()          # Auto-categorization
    - spending_by_shop()              # Shop analysis
    - payment_mode_distribution()     # Payment breakdown
    - spending_trend()                # Time series
    - top_items()                     # Item frequency
    - spending_by_category()          # Category treemap
```

**Features:**
- 8 predefined categories (customizable)
- Interactive Plotly visualizations
- Trend analysis with cumulative spending
- Export functionality

## 🎯 Usage Examples

### Upload Receipt
```python
# Staff workflow:
1. Click "Upload Receipt"
2. Select image file
3. Click "Process Receipt"
4. Verify extracted data
5. Edit if needed
6. Click "Save to Database"
```

### Query Database
```python
# Example queries:
"Show all receipts from Starbucks"
"What did I spend on office supplies last quarter?"
"Find cash purchases over $50"
"Total spending in January 2024"
"Which payment method do I use most?"
```

### View Analytics
```python
# Management dashboard shows:
- Total spending across all receipts
- Spending by shop (bar chart)
- Payment mode distribution (pie chart)
- Spending trend over time (line chart)
- Top 10 purchased items
- Category breakdown (treemap)
- Exportable CSV reports
```

## 🔧 Configuration

### Environment Variables
```bash
# .env
OPENAI_API_KEY=sk-your-key-here          # Required
TESSERACT_PATH=/usr/local/bin/tesseract  # Optional
```

### Customizing Categories
Edit `analytics.py`:
```python
self.categories = {
    'Your Category': ['keyword1', 'keyword2', 'keyword3'],
    'Office Supplies': ['office', 'stationery', 'printer'],
    # Add more categories...
}
```

### Switching to Vision API
In `app.py`:
```python
from ocr_processor import OCRProcessorVision as OCRProcessor
```
More accurate but higher API costs (~$0.01-0.03 per image).

## 🌐 Deployment

### Streamlit Community Cloud (Free)
```bash
# 1. Push to GitHub
git add . && git commit -m "Deploy" && git push

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Add secrets: OPENAI_API_KEY
# 5. Add packages.txt: echo "tesseract-ocr" > packages.txt
```

### Docker
```dockerfile
docker build -t receipt-verification .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key receipt-verification
```

### Heroku
```bash
heroku create your-app
heroku buildpacks:add heroku-community/apt
heroku buildpacks:add heroku/python
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

## 📊 Technical Details

### Data Schema
```python
{
    'Shop Name': str,
    'Date of Purchase': str (YYYY-MM-DD),
    'Item Purchased': str,
    'Mode of Purchase': str (Cash/Credit/Debit/E-Wallet/Other),
    'Unit Purchased': int,
    'Unit Price': float,
    'Total Price': float,
    'Image Path': str
}
```

### API Usage
- **Embeddings**: ~$0.00002 per receipt
- **GPT-4 Mini**: ~$0.0001-0.0005 per receipt
- **Vision API** (optional): ~$0.01-0.03 per receipt

### Performance
- OCR processing: 1-3 seconds per receipt
- Query response: 1-2 seconds
- Analytics rendering: <1 second
- Supports thousands of receipts

## 🧪 Testing

Run comprehensive test suite:
```bash
python test_system.py
```

Tests include:
- ✅ Import verification
- ✅ Tesseract installation
- ✅ OpenAI API connectivity
- ✅ ChromaDB functionality
- ✅ Directory structure
- ✅ OCR processor
- ✅ RAG chatbot
- ✅ Analytics module

## 🛡️ Security

- Environment variables for API keys
- No hardcoded credentials
- `.gitignore` for sensitive files
- Input validation on all uploads
- Sanitized database queries

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [OpenAI](https://openai.com/) - GPT-4 and embeddings
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Plotly](https://plotly.com/) - Interactive visualizations

## 📧 Support

For issues and questions:
- 🐛 [Open an issue](https://github.com/yourusername/receipt-verification/issues)
- 💬 [Start a discussion](https://github.com/yourusername/receipt-verification/discussions)
- 📧 Email: your-email@example.com

## 🗺️ Roadmap

- [ ] Multi-user authentication
- [ ] PostgreSQL database backend
- [ ] Email report automation
- [ ] Mobile app (iOS/Android)
- [ ] Bulk upload capability
- [ ] Budget alert notifications
- [ ] Multi-language support
- [ ] Receipt image preprocessing
- [ ] API endpoints for integration
- [ ] Advanced fraud detection

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/receipt-verification&type=Date)](https://star-history.com/#yourusername/receipt-verification&Date)

---

**Made with ❤️ for accounts officers and finance teams**

If you find this useful, please ⭐ star the repository!
