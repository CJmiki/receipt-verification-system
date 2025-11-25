"""
RAG Chatbot for Receipt Query System + HR Guidelines
Handles PDF loading more reliably
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
    """
    
    def __init__(self, hr_guidelines_path=None):
        """Initialize RAG components"""
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.client = OpenAI(api_key=api_key)
            
            # Initialize ChromaDB
            try:
                self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
                print("✅ ChromaDB persistent client initialized")
            except Exception as e:
                print(f"Warning: ChromaDB persistent client failed: {e}")
                self.chroma_client = chromadb.Client()
                print("✅ Using in-memory ChromaDB client")
            
            # Get or create collection for receipts
            try:
                self.receipts_collection = self.chroma_client.get_collection(name="receipts")
                print(f"✅ Loaded existing receipts collection with {self.receipts_collection.count()} receipts")
            except:
                self.receipts_collection = self.chroma_client.create_collection(
                    name="receipts",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ Created new receipts collection")
            
            # Handle HR guidelines collection
            self._initialize_hr_collection(hr_guidelines_path)
                
        except Exception as e:
            print(f"RAGChatbot initialization error: {e}")
            raise
    
    def _initialize_hr_collection(self, hr_guidelines_path):
        """Initialize HR collection and load PDF if needed"""
        try:
            # Try to get existing collection
            self.hr_collection = self.chroma_client.get_collection(name="hr_guidelines")
            existing_count = self.hr_collection.count()
            
            if existing_count > 0:
                print(f"✅ Loaded existing HR guidelines collection with {existing_count} sections")
                return
            else:
                print("⚠️ HR guidelines collection exists but is empty, will reload...")
                # Delete and recreate
                self.chroma_client.delete_collection(name="hr_guidelines")
                raise Exception("Empty collection")
                
        except:
            # Create new collection
            self.hr_collection = self.chroma_client.create_collection(
                name="hr_guidelines",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new HR guidelines collection")
            
            # Auto-detect PDF location
            if hr_guidelines_path is None:
                possible_paths = [
                    "HR Guidelines.pdf",
                    "./HR Guidelines.pdf",
                    "data/HR Guidelines.pdf",
                    "../HR Guidelines.pdf",
                    "pages/HR Guidelines.pdf"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        hr_guidelines_path = path
                        print(f"📄 Found HR Guidelines at: {path}")
                        break
            
            # Load PDF if found
            if hr_guidelines_path and os.path.exists(hr_guidelines_path):
                print(f"📖 Loading HR Guidelines from: {hr_guidelines_path}")
                success = self.load_hr_guidelines(hr_guidelines_path)
                if success:
                    print(f"✅ Successfully loaded HR Guidelines!")
                else:
                    print(f"⚠️ Failed to load HR Guidelines")
            else:
                print(f"⚠️ HR Guidelines PDF not found!")
                print(f"   Searched in: {', '.join(possible_paths)}")
                print(f"   Current directory: {os.getcwd()}")
                print(f"   📌 Place 'HR Guidelines.pdf' in the project root folder")
    
    def load_hr_guidelines(self, pdf_path):
        """Load HR guidelines PDF and add to vector database"""
        try:
            if not os.path.exists(pdf_path):
                print(f"❌ File not found: {pdf_path}")
                return False
            
            print(f"📖 Extracting text from PDF...")
            pdf_text = self._extract_pdf_text(pdf_path)
            
            if not pdf_text or len(pdf_text.strip()) < 100:
                print(f"❌ PDF appears empty or unreadable")
                print(f"   Extracted only {len(pdf_text)} characters")
                
                # Try alternative extraction method
                print("🔄 Trying alternative extraction method...")
                pdf_text = self._extract_pdf_text_alternative(pdf_path)
                
                if not pdf_text or len(pdf_text.strip()) < 100:
                    print("❌ Alternative method also failed")
                    return False
            
            print(f"✅ Extracted {len(pdf_text)} characters from PDF")
            
            # Split into chunks
            chunks = self._split_text_into_chunks(pdf_text, chunk_size=1000, overlap=200)
            print(f"📄 Split into {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print("❌ No chunks created from PDF")
                return False
            
            # Add each chunk to vector database
            success_count = 0
            failed_count = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                doc_id = f"hr_guideline_chunk_{i}"
                
                try:
                    # Get embedding
                    embedding = self._get_embedding(chunk)
                    
                    if embedding is None:
                        failed_count += 1
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
                    
                    # Progress indicator every 5 chunks
                    if (i + 1) % 5 == 0:
                        print(f"   Processed {i + 1}/{len(chunks)} chunks...")
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to process chunk {i}: {e}")
                    failed_count += 1
            
            if success_count > 0:
                print(f"✅ Successfully loaded {success_count}/{len(chunks)} chunks")
                print(f"   Failed: {failed_count}")
                return True
            else:
                print(f"❌ Failed to load any chunks")
                return False
            
        except Exception as e:
            print(f"❌ Error loading HR guidelines: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        try:
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
                    except Exception as e:
                        print(f"   ⚠️ Page {i+1} extraction failed: {e}")
                        continue
            
            return text.strip()
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    
    def _extract_pdf_text_alternative(self, pdf_path):
        """Alternative PDF extraction method"""
        try:
            import io
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    try:
                        # Try different extraction methods
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except:
                        continue
            
            return text.strip()
        except:
            return ""
    
    def _split_text_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:  # Only add substantial chunks
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _get_embedding(self, text):
        """Generate embedding vector using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def add_receipt(self, record):
        """Add a receipt record to the vector database"""
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
            
            doc_id = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            embedding = self._get_embedding(doc_text)
            
            if embedding is None:
                return False
            
            metadata = {k: str(v) for k, v in record.items()}
            
            self.receipts_collection.add(
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding receipt: {e}")
            return False
    
    def query(self, question, df=None):
        """Query the system using RAG"""
        try:
            query_type = self._determine_query_type(question)
            
            if query_type == 'hr_policy':
                return self._query_hr_guidelines(question)
            elif query_type == 'receipts':
                return self._query_receipts(question, df)
            else:
                return self._query_mixed(question, df)
            
        except Exception as e:
            print(f"Query error: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'relevant_receipts': None,
                'query_type': 'error'
            }
    
    def _determine_query_type(self, question):
        """Determine query type"""
        question_lower = question.lower()
        
        hr_keywords = [
            'policy', 'policies', 'guideline', 'guidelines', 'petty cash',
            'reimbursement', 'claim', 'allowable', 'allowed', 'approval',
            'limit', 'maximum', 'minimum', 'rules', 'regulation', 'procedure',
            'how to', 'what is allowed', 'can i', 'is it allowed'
        ]
        
        receipt_keywords = [
            'bought', 'purchased', 'spent', 'total', 'shop', 'store',
            'last month', 'show me', 'find', 'receipts'
        ]
        
        hr_count = sum(1 for kw in hr_keywords if kw in question_lower)
        receipt_count = sum(1 for kw in receipt_keywords if kw in question_lower)
        
        if hr_count > 0 and receipt_count == 0:
            return 'hr_policy'
        elif receipt_count > 0 and hr_count == 0:
            return 'receipts'
        elif hr_count > 0 and receipt_count > 0:
            return 'mixed'
        else:
            return 'receipts'
    
    def _query_hr_guidelines(self, question):
        """Query HR guidelines"""
        try:
            collection_count = self.hr_collection.count()
            
            if collection_count == 0:
                return {
                    'answer': "❌ HR guidelines are not loaded. Please ensure the HR Guidelines PDF is in the project folder and restart the application.",
                    'relevant_receipts': None,
                    'query_type': 'hr_policy'
                }
            
            query_embedding = self._get_embedding(question)
            if query_embedding is None:
                return {
                    'answer': "Sorry, I couldn't process your question.",
                    'relevant_receipts': None,
                    'query_type': 'hr_policy'
                }
            
            n_results = min(3, collection_count)
            results = self.hr_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            context = ""
            if results['documents'] and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
            
            answer = self._generate_hr_answer(question, context)
            
            return {
                'answer': answer,
                'relevant_receipts': None,
                'query_type': 'hr_policy'
            }
            
        except Exception as e:
            print(f"HR query error: {e}")
            return {
                'answer': f"Error querying HR guidelines: {str(e)}",
                'relevant_receipts': None,
                'query_type': 'hr_policy'
            }
    
    def _generate_hr_answer(self, question, context):
        """Generate answer for HR policy questions"""
        try:
            if not context or len(context.strip()) < 10:
                return "I couldn't find relevant information in the HR guidelines for your question. Please try rephrasing or ask about specific policies like petty cash limits or reimbursement procedures."
            
            prompt = f"""You are a helpful HR assistant answering questions about company policies.

RELEVANT POLICY SECTIONS:
{context}

USER QUESTION: {question}

Instructions:
- Answer based ONLY on the provided policy sections
- Be clear, concise, and professional
- If the policy doesn't cover the question, say so honestly
- Use bullet points for clarity when listing requirements
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an HR policy assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "I'm sorry, I couldn't generate an answer."
    
    def _query_receipts(self, question, df):
        """Query receipt database"""
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
                    'answer': "Receipt database is empty. Please upload receipts first.",
                    'relevant_receipts': None,
                    'query_type': 'receipts'
                }
            
            query_embedding = self._get_embedding(question)
            if query_embedding is None:
                return {
                    'answer': "Sorry, I couldn't process your question.",
                    'relevant_receipts': None,
                    'query_type': 'receipts'
                }
            
            n_results = min(5, collection_count)
            results = self.receipts_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            relevant_docs = []
            relevant_indices = []
            
            if results['documents'] and len(results['documents'][0]) > 0:
                relevant_docs = results['documents'][0]
                
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
                        except:
                            continue
            
            relevant_df = df.iloc[relevant_indices].drop_duplicates() if relevant_indices else pd.DataFrame()
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
                'answer': f"Error querying receipts: {str(e)}",
                'relevant_receipts': None,
                'query_type': 'receipts'
            }
    
    def _generate_receipt_answer(self, question, context, df):
        """Generate answer for receipt queries"""
        try:
            total_receipts = len(df)
            total_spending = df['Total Price'].sum()
            
            stats = f"Total Receipts: {total_receipts}, Total Spending: ${total_spending:.2f}"
            
            prompt = f"""You are a helpful assistant analyzing receipt data.

DATABASE STATISTICS: {stats}

RELEVANT RECEIPTS:
{context}

USER QUESTION: {question}

Provide a clear answer based on the data."""
            
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
        """Handle mixed queries"""
        hr_result = self._query_hr_guidelines(question)
        receipt_result = self._query_receipts(question, df) if df is not None else None
        
        combined_answer = f"""**Policy Information:**
{hr_result['answer']}

**Your Receipt Data:**
{receipt_result['answer'] if receipt_result else 'No receipt data available.'}"""
        
        return {
            'answer': combined_answer,
            'relevant_receipts': receipt_result['relevant_receipts'] if receipt_result else None,
            'query_type': 'mixed'
        }
    
    def get_stats(self):
        """Get statistics"""
        try:
            return {
                'total_receipts': self.receipts_collection.count(),
                'total_hr_chunks': self.hr_collection.count()
            }
        except:
            return {'total_receipts': 0, 'total_hr_chunks': 0}


# Test function
if __name__ == "__main__":
    print("Testing RAG Chatbot...")
    chatbot = RAGChatbot(hr_guidelines_path="HR Guidelines.pdf")
    stats = chatbot.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    if stats['total_hr_chunks'] > 0:
        print("\n✅ HR Guidelines loaded successfully!")
    else:
        print("\n❌ HR Guidelines not loaded")

