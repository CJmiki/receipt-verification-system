"""
RAG Chatbot for Receipt Query System + HR Guidelines
Uses ChromaDB for vector storage and OpenAI for embeddings and responses
"""

import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2

load_dotenv()

class RAGChatbot:
    """
    Retrieval-Augmented Generation Chatbot for:
    1. Querying receipt database
    2. Answering HR policy questions (petty cash guidelines)
    
    Features:
    - Semantic search using OpenAI embeddings
    - ChromaDB vector storage
    - Context-aware responses with GPT-4
    - Natural language query interface
    - HR guidelines knowledge base
    """
    
    def __init__(self, hr_guidelines_path=None):
        """Initialize RAG components"""
        # Auto-detect HR Guidelines PDF location
        if hr_guidelines_path is None:
            # Try common locations
            possible_paths = [
                "HR Guidelines.pdf",
                "./HR Guidelines.pdf",
                "data/HR Guidelines.pdf",
                "../HR Guidelines.pdf"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    hr_guidelines_path = path
                    print(f"Found HR Guidelines at: {path}")
                    break
        """Initialize RAG components"""
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.client = OpenAI(api_key=api_key)
            
            # Initialize ChromaDB with error handling
            try:
                # Try persistent client first
                self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
            except Exception as e:
                print(f"Warning: ChromaDB persistent client failed: {e}")
                print("Falling back to in-memory client...")
                # Fallback to in-memory client
                self.chroma_client = chromadb.Client()
            
            # Get or create collection for receipts
            try:
                self.receipts_collection = self.chroma_client.get_collection(name="receipts")
                print(f"Loaded existing receipts collection with {self.receipts_collection.count()} receipts")
            except:
                self.receipts_collection = self.chroma_client.create_collection(
                    name="receipts",
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                print("Created new receipts collection")
            
            # Get or create collection for HR guidelines
            try:
                self.hr_collection = self.chroma_client.get_collection(name="hr_guidelines")
                print(f"✅ Loaded existing HR guidelines collection with {self.hr_collection.count()} sections")
            except:
                self.hr_collection = self.chroma_client.create_collection(
                    name="hr_guidelines",
                    metadata={"hnsw:space": "cosine"}
                )
                print("Created new HR guidelines collection")
                
                # Load HR guidelines if file exists
                if hr_guidelines_path and os.path.exists(hr_guidelines_path):
                    print(f"📄 Loading HR Guidelines from: {hr_guidelines_path}")
                    self.load_hr_guidelines(hr_guidelines_path)
                else:
                    print(f"⚠️ WARNING: HR Guidelines PDF not found!")
                    print(f"   Searched for: {hr_guidelines_path if hr_guidelines_path else 'No path specified'}")
                    print(f"   Current directory: {os.getcwd()}")
                    print(f"   Place 'HR Guidelines.pdf' in the same folder as app.py")
                
        except Exception as e:
            print(f"RAGChatbot initialization error: {e}")
            raise
    
    def load_hr_guidelines(self, pdf_path):
        """
        Load HR guidelines PDF and add to vector database
        
        Args:
            pdf_path (str): Path to HR Guidelines PDF
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"📖 Loading HR Guidelines from {pdf_path}...")
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                print(f"❌ File not found: {pdf_path}")
                return False
            
            # Extract text from PDF
            pdf_text = self._extract_pdf_text(pdf_path)
            
            if not pdf_text or len(pdf_text.strip()) < 100:
                print("❌ ERROR: PDF appears to be empty or unreadable")
                print(f"   Extracted text length: {len(pdf_text)} characters")
                print(f"   Preview: '{pdf_text[:200]}'")
                print("\n💡 TROUBLESHOOTING:")
                print("   1. Check if PDF is text-based (not a scanned image)")
                print("   2. Try opening the PDF manually to verify content")
                print("   3. If it's a scanned PDF, you'll need OCR processing")
                print("   4. Try re-saving the PDF with 'Save As Text' option")
                return False
            
            print(f"✅ Successfully extracted {len(pdf_text)} characters from PDF")
            
            # Split text into chunks for better retrieval
            chunks = self._split_text_into_chunks(pdf_text, chunk_size=1000, overlap=200)
            
            print(f"📄 Split HR guidelines into {len(chunks)} chunks")
            
            # Add each chunk to vector database
            success_count = 0
            for i, chunk in enumerate(chunks):
                doc_id = f"hr_guideline_chunk_{i}"
                
                # Get embedding
                embedding = self._get_embedding(chunk)
                
                if embedding is None:
                    print(f"⚠️ Failed to generate embedding for chunk {i}")
                    continue
                
                # Add to collection
                self.hr_collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": "HR Guidelines",
                        "chunk_id": str(i),
                        "type": "policy"
                    }],
                    ids=[doc_id]
                )
                success_count += 1
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"   Processed {i + 1}/{len(chunks)} chunks...")
            
            print(f"✅ Successfully loaded {success_count}/{len(chunks)} chunks from HR Guidelines")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error loading HR guidelines: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def _extract_pdf_text(self, pdf_path):
        """
        Extract text from PDF file with enhanced error handling
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            str: Extracted text
        """
        try:
            print(f"📄 Opening PDF: {pdf_path}")
            print(f"   File size: {os.path.getsize(pdf_path)} bytes")
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"   PDF has {num_pages} pages")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            print(f"   ✓ Page {i+1}: Extracted {len(page_text)} characters")
                        else:
                            print(f"   ⚠ Page {i+1}: No text found (might be scanned image)")
                    except Exception as page_error:
                        print(f"   ✗ Page {i+1}: Error - {page_error}")
                        continue
            
            print(f"   Total text extracted: {len(text)} characters")
            
            if len(text.strip()) < 100:
                print(f"   ⚠️ WARNING: Very little text extracted!")
                print(f"   This PDF might be:")
                print(f"      - A scanned document (needs OCR)")
                print(f"      - Protected/encrypted")
                print(f"      - Empty or corrupted")
                print(f"   First 200 chars: {text[:200]}")
            
            return text
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return ""
    
    def _split_text_into_chunks(self, text, chunk_size=1000, overlap=200):
        """
        Split text into overlapping chunks for better retrieval
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum characters per chunk
            overlap (int): Characters to overlap between chunks
        
        Returns:
            list: List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for period, newline, or other sentence endings
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def add_receipt(self, record):
        """
        Add a receipt record to the vector database
        
        Args:
            record (dict): Receipt data with keys:
                - Shop Name
                - Date of Purchase
                - Item Purchased
                - Mode of Purchase
                - Unit Purchased
                - Unit Price
                - Total Price
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create document text for embedding
            doc_text = f"""
Shop: {record.get('Shop Name', '')}
Date: {record.get('Date of Purchase', '')}
Item: {record.get('Item Purchased', '')}
Payment Mode: {record.get('Mode of Purchase', '')}
Units: {record.get('Unit Purchased', 0)}
Unit Price: ${record.get('Unit Price', 0)}
Total: ${record.get('Total Price', 0)}
"""
            
            # Generate unique ID with timestamp
            doc_id = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Get embedding from OpenAI
            embedding = self._get_embedding(doc_text)
            
            if embedding is None:
                print("Failed to generate embedding")
                return False
            
            # Convert all metadata values to strings (ChromaDB requirement)
            metadata = {k: str(v) for k, v in record.items()}
            
            # Add to ChromaDB collection
            self.receipts_collection.add(
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            print(f"✅ Added receipt: {record.get('Shop Name', 'Unknown')} - ${record.get('Total Price', 0)}")
            return True
            
        except Exception as e:
            print(f"Error adding receipt to vector DB: {e}")
            return False
    
    def _get_embedding(self, text):
        """
        Generate embedding vector for text using OpenAI
        
        Args:
            text (str): Text to embed
        
        Returns:
            list: Embedding vector or None if failed
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # 1536 dimensions
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return None
    
    def query(self, question, df=None):
        """
        Query the system using RAG - handles both receipt queries and HR policy questions
        
        Args:
            question (str): Natural language question
            df (DataFrame): Complete receipts dataframe (optional, for receipt queries)
        
        Returns:
            dict: {
                'answer': str - Generated answer,
                'relevant_receipts': DataFrame - Relevant receipts or None,
                'query_type': str - 'receipts', 'hr_policy', or 'mixed'
            }
        """
        try:
            # Determine query type based on question content
            query_type = self._determine_query_type(question)
            
            print(f"Query type detected: {query_type}")
            
            if query_type == 'hr_policy':
                return self._query_hr_guidelines(question)
            elif query_type == 'receipts':
                return self._query_receipts(question, df)
            else:  # mixed
                return self._query_mixed(question, df)
            
        except Exception as e:
            print(f"Query processing error: {e}")
            return {
                'answer': f"I encountered an error processing your question. Please try again.",
                'relevant_receipts': None,
                'query_type': 'error'
            }
    
    def _determine_query_type(self, question):
        """
        Determine if question is about HR policies or receipts
        
        Args:
            question (str): User question
        
        Returns:
            str: 'hr_policy', 'receipts', or 'mixed'
        """
        question_lower = question.lower()
        
        # HR policy keywords
        hr_keywords = [
            'policy', 'policies', 'guideline', 'guidelines', 'petty cash',
            'reimbursement', 'claim', 'allowable', 'allowed', 'approval',
            'limit', 'maximum', 'minimum', 'rules', 'regulation', 'procedure',
            'how to', 'what is allowed', 'can i', 'is it allowed', 'compliance'
        ]
        
        # Receipt query keywords
        receipt_keywords = [
            'bought', 'purchased', 'spent', 'total', 'shop', 'store',
            'last month', 'january', 'february', 'march', 'april',
            'how much', 'show me', 'find', 'receipts from'
        ]
        
        hr_count = sum(1 for keyword in hr_keywords if keyword in question_lower)
        receipt_count = sum(1 for keyword in receipt_keywords if keyword in question_lower)
        
        if hr_count > 0 and receipt_count == 0:
            return 'hr_policy'
        elif receipt_count > 0 and hr_count == 0:
            return 'receipts'
        elif hr_count > 0 and receipt_count > 0:
            return 'mixed'
        else:
            # Default to receipts if unclear
            return 'receipts'
    
    def _query_hr_guidelines(self, question):
        """
        Query HR guidelines/policies
        
        Args:
            question (str): Policy question
        
        Returns:
            dict: Response with answer
        """
        try:
            collection_count = self.hr_collection.count()
            
            if collection_count == 0:
                return {
                    'answer': "HR guidelines are not loaded. Please ensure the HR Guidelines PDF is available.",
                    'relevant_receipts': None,
                    'query_type': 'hr_policy'
                }
            
            print(f"Querying {collection_count} HR guideline chunks...")
            
            # Get embedding for question
            query_embedding = self._get_embedding(question)
            
            if query_embedding is None:
                return {
                    'answer': "Sorry, I couldn't process your question.",
                    'relevant_receipts': None,
                    'query_type': 'hr_policy'
                }
            
            # Search HR guidelines
            n_results = min(3, collection_count)
            results = self.hr_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Extract relevant context
            context = ""
            if results['documents'] and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
                print(f"Found {len(results['documents'][0])} relevant guideline sections")
            
            # Generate answer
            answer = self._generate_hr_answer(question, context)
            
            return {
                'answer': answer,
                'relevant_receipts': None,
                'query_type': 'hr_policy'
            }
            
        except Exception as e:
            print(f"HR query error: {e}")
            return {
                'answer': "I encountered an error accessing HR guidelines.",
                'relevant_receipts': None,
                'query_type': 'hr_policy'
            }
    
    def _generate_hr_answer(self, question, context):
        """
        Generate answer for HR policy questions
        
        Args:
            question (str): User's question
            context (str): Retrieved policy context
        
        Returns:
            str: Generated answer
        """
        try:
            prompt = f"""
You are a helpful HR assistant answering questions about company policies, specifically petty cash and reimbursement guidelines.

RELEVANT POLICY SECTIONS:
{context}

USER QUESTION: {question}

Instructions:
- Answer based ONLY on the provided policy sections
- Be clear, concise, and professional
- If the policy doesn't cover the question, say so honestly
- Cite specific policy sections when relevant
- Use bullet points for clarity when listing requirements or rules
- Be helpful and provide actionable guidance
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HR policy assistant. Provide accurate answers based on company guidelines. Be clear about policy requirements and restrictions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Lower for more factual responses
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"HR answer generation error: {e}")
            return "I'm sorry, I couldn't generate an answer at this time."
    
    def _query_receipts(self, question, df):
        """
        Query receipt database (original functionality)
        
        Args:
            question (str): Receipt query
            df (DataFrame): Receipts dataframe
        
        Returns:
            dict: Response with answer and relevant receipts
        """
        try:
            if df is None or len(df) == 0:
                return {
                    'answer': "No receipts in database. Please upload some receipts first.",
                    'relevant_receipts': None,
                    'query_type': 'receipts'
                }
            
            collection_count = self.receipts_collection.count()
            
            if collection_count == 0:
                return {
                    'answer': "The receipt database is empty. Please upload some receipts first.",
                    'relevant_receipts': None,
                    'query_type': 'receipts'
                }
            
            print(f"Querying {collection_count} receipts...")
            
            # Get embedding for question
            query_embedding = self._get_embedding(question)
            
            if query_embedding is None:
                return {
                    'answer': "Sorry, I couldn't process your question.",
                    'relevant_receipts': None,
                    'query_type': 'receipts'
                }
            
            # Search receipt database
            n_results = min(5, collection_count)
            results = self.receipts_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Extract relevant documents and metadata
            relevant_docs = []
            relevant_indices = []
            
            if results['documents'] and len(results['documents'][0]) > 0:
                relevant_docs = results['documents'][0]
                print(f"Found {len(relevant_docs)} relevant receipts")
                
                # Match metadata to dataframe rows
                if results['metadatas'] and len(results['metadatas'][0]) > 0:
                    for metadata in results['metadatas'][0]:
                        try:
                            mask = (
                                (df['Shop Name'].astype(str) == str(metadata.get('Shop Name', ''))) &
                                (df['Date of Purchase'].astype(str) == str(metadata.get('Date of Purchase', ''))) &
                                (df['Item Purchased'].astype(str) == str(metadata.get('Item Purchased', '')))
                            )
                            matching_indices = df[mask].index.tolist()
                            relevant_indices.extend(matching_indices)
                        except Exception as e:
                            print(f"Error matching metadata: {e}")
                            continue
            
            # Create relevant receipts dataframe
            if relevant_indices:
                relevant_df = df.iloc[relevant_indices].drop_duplicates()
            else:
                relevant_df = pd.DataFrame()
            
            # Generate answer
            context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant receipts found."
            answer = self._generate_receipt_answer(question, context, df)
            
            return {
                'answer': answer,
                'relevant_receipts': relevant_df if not relevant_df.empty else None,
                'query_type': 'receipts'
            }
            
        except Exception as e:
            print(f"Receipt query error: {e}")
            return {
                'answer': "I encountered an error querying receipts.",
                'relevant_receipts': None,
                'query_type': 'receipts'
            }
    
    def _generate_receipt_answer(self, question, context, df):
        """
        Generate answer for receipt queries (original method)
        """
        try:
            # Create summary statistics
            total_receipts = len(df)
            total_spending = df['Total Price'].sum()
            date_min = df['Date of Purchase'].min()
            date_max = df['Date of Purchase'].max()
            unique_shops = df['Shop Name'].nunique()
            
            stats = f"""
Total Receipts: {total_receipts}
Total Spending: ${total_spending:.2f}
Date Range: {date_min} to {date_max}
Unique Shops: {unique_shops}
"""
            
            prompt = f"""
You are a helpful assistant analyzing receipt data for expense tracking.

DATABASE STATISTICS:
{stats}

RELEVANT RECEIPTS:
{context}

USER QUESTION: {question}

Instructions:
- Provide a clear answer based on the data
- Format currency as $X.XX
- Be conversational and professional
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a receipt analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "I'm sorry, I couldn't generate an answer."
    
    def _query_mixed(self, question, df):
        """
        Handle queries that involve both HR policies and receipts
        
        Args:
            question (str): Mixed query
            df (DataFrame): Receipts dataframe
        
        Returns:
            dict: Combined response
        """
        # Query both sources
        hr_result = self._query_hr_guidelines(question)
        receipt_result = self._query_receipts(question, df) if df is not None else None
        
        # Combine answers
        combined_answer = f"""
**Policy Information:**
{hr_result['answer']}

**Your Receipt Data:**
{receipt_result['answer'] if receipt_result else 'No receipt data available.'}
"""
        
        return {
            'answer': combined_answer,
            'relevant_receipts': receipt_result['relevant_receipts'] if receipt_result else None,
            'query_type': 'mixed'
        }
    
    def rebuild_index(self, csv_path="data/receipts.csv"):
        """
        Rebuild the receipt vector database from CSV file
        """
        try:
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                return False
            
            print("Rebuilding receipt index from CSV...")
            
            # Delete existing collection
            try:
                self.chroma_client.delete_collection(name="receipts")
                print("Deleted old receipts collection")
            except:
                print("No existing receipts collection to delete")
            
            # Create new collection
            self.receipts_collection = self.chroma_client.create_collection(
                name="receipts",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} receipts from CSV")
            
            # Add each receipt
            success_count = 0
            for idx, row in df.iterrows():
                record = row.to_dict()
                if self.add_receipt(record):
                    success_count += 1
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} receipts...")
            
            print(f"✅ Rebuilt receipt index with {success_count}/{len(df)} receipts")
            return True
            
        except Exception as e:
            print(f"Rebuild error: {e}")
            return False
    
    def get_stats(self):
        """Get statistics about the vector databases"""
        try:
            return {
                'total_receipts': self.receipts_collection.count(),
                'total_hr_chunks': self.hr_collection.count(),
                'receipts_collection': self.receipts_collection.name,
                'hr_collection': self.hr_collection.name
            }
        except:
            return {
                'total_receipts': 0,
                'total_hr_chunks': 0,
                'receipts_collection': 'receipts',
                'hr_collection': 'hr_guidelines'
            }


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = RAGChatbot(hr_guidelines_path="HR Guidelines.pdf")
    
    # Test HR policy query
    print("\n=== HR Policy Query ===")
    result = chatbot.query("What is the petty cash reimbursement limit?")
    print(f"Answer: {result['answer']}")
    
    # Test receipt query
    print("\n=== Receipt Query ===")
    sample_receipt = {
        'Shop Name': 'Office Depot',
        'Date of Purchase': '2024-01-15',
        'Item Purchased': 'Stationery',
        'Mode of Purchase': 'Cash',
        'Unit Purchased': 1,
        'Unit Price': 25.00,
        'Total Price': 25.00,
        'Image Path': 'test.jpg'
    }
    chatbot.add_receipt(sample_receipt)
    
    sample_df = pd.DataFrame([sample_receipt])
    result = chatbot.query("Show me office supply purchases", sample_df)
    print(f"Answer: {result['answer']}")

