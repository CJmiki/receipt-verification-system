"""
RAG Chatbot for Receipt Query System
Uses ChromaDB for vector storage and OpenAI for embeddings and responses
"""

import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class RAGChatbot:
    """
    Retrieval-Augmented Generation Chatbot for querying receipt database
    
    Features:
    - Semantic search using OpenAI embeddings
    - ChromaDB vector storage
    - Context-aware responses with GPT-4
    - Natural language query interface
    """
    
    def __init__(self):
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
                self.collection = self.chroma_client.get_collection(name="receipts")
                print(f"Loaded existing collection with {self.collection.count()} receipts")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="receipts",
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                print("Created new receipts collection")
                
        except Exception as e:
            print(f"RAGChatbot initialization error: {e}")
            raise
    
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
            self.collection.add(
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
    
    def query(self, question, df):
        """
        Query the receipt database using RAG
        
        Args:
            question (str): Natural language question
            df (DataFrame): Complete receipts dataframe
        
        Returns:
            dict: {
                'answer': str - Generated answer,
                'relevant_receipts': DataFrame - Relevant receipts or None
            }
        """
        try:
            # Check if collection has any data
            collection_count = self.collection.count()
            
            if collection_count == 0:
                return {
                    'answer': "The database is empty. Please upload some receipts first.",
                    'relevant_receipts': None
                }
            
            print(f"Querying {collection_count} receipts...")
            
            # Get embedding for the question
            query_embedding = self._get_embedding(question)
            
            if query_embedding is None:
                return {
                    'answer': "Sorry, I couldn't process your question. Please try again.",
                    'relevant_receipts': None
                }
            
            # Search vector database for similar receipts
            n_results = min(5, collection_count)  # Get top 5 or all if less
            results = self.collection.query(
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
                            # Find matching row in dataframe
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
            
            # Generate answer using GPT with context
            context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant receipts found."
            answer = self._generate_answer(question, context, df)
            
            return {
                'answer': answer,
                'relevant_receipts': relevant_df if not relevant_df.empty else None
            }
            
        except Exception as e:
            print(f"Query processing error: {e}")
            return {
                'answer': f"I encountered an error processing your question. Please try again or rephrase your query.",
                'relevant_receipts': None
            }
    
    def _generate_answer(self, question, context, df):
        """
        Generate natural language answer using GPT-4
        
        Args:
            question (str): User's question
            context (str): Retrieved receipt context
            df (DataFrame): Complete receipts dataframe for statistics
        
        Returns:
            str: Generated answer
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
            
            # Create prompt for GPT
            prompt = f"""
You are a helpful assistant analyzing receipt data for a business expense tracking system.
Answer the user's question based on the provided context and statistics.

DATABASE STATISTICS:
{stats}

RELEVANT RECEIPTS FROM VECTOR SEARCH:
{context}

USER QUESTION: {question}

Instructions:
- Provide a clear, concise answer based on the data
- If you mention specific amounts or receipts, reference them naturally
- If the data doesn't contain information to answer the question, say so politely
- Be helpful and professional
- Format currency as $X.XX
- Use natural language, not technical jargon
"""
            
            # Call GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful receipt analysis assistant. Provide clear, accurate answers based on the data provided. Be conversational and professional."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            return "I'm sorry, I couldn't generate an answer at this time. Please try again."
    
    def rebuild_index(self, csv_path="data/receipts.csv"):
        """
        Rebuild the entire vector database from CSV file
        Useful after bulk data imports or corruption
        
        Args:
            csv_path (str): Path to receipts CSV file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                return False
            
            print("Rebuilding vector index from CSV...")
            
            # Delete existing collection
            try:
                self.chroma_client.delete_collection(name="receipts")
                print("Deleted old collection")
            except:
                print("No existing collection to delete")
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="receipts",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} receipts from CSV")
            
            # Add each receipt to vector database
            success_count = 0
            for idx, row in df.iterrows():
                record = row.to_dict()
                if self.add_receipt(record):
                    success_count += 1
                
                # Progress indicator
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} receipts...")
            
            print(f"✅ Rebuilt index with {success_count}/{len(df)} receipts")
            return True
            
        except Exception as e:
            print(f"Rebuild error: {e}")
            return False
    
    def get_stats(self):
        """
        Get statistics about the vector database
        
        Returns:
            dict: Statistics including count, etc.
        """
        try:
            return {
                'total_receipts': self.collection.count(),
                'collection_name': self.collection.name
            }
        except:
            return {
                'total_receipts': 0,
                'collection_name': 'receipts'
            }


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Add sample receipt
    sample_receipt = {
        'Shop Name': 'Walmart',
        'Date of Purchase': '2024-01-15',
        'Item Purchased': 'Office Supplies',
        'Mode of Purchase': 'Credit Card',
        'Unit Purchased': 1,
        'Unit Price': 45.99,
        'Total Price': 45.99,
        'Image Path': 'test.jpg'
    }
    
    chatbot.add_receipt(sample_receipt)
    
    # Query example
    sample_df = pd.DataFrame([sample_receipt])
    result = chatbot.query("What did I buy at Walmart?", sample_df)
    print(f"\nAnswer: {result['answer']}")