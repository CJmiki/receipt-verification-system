import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ocr_processor import OCRProcessorVision as OCRProcessor
from rag_chatbot import RAGChatbot
from analytics import ReceiptAnalytics

# Page config
st.set_page_config(page_title="Receipt Verification System", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    ocr = OCRProcessor()
    rag = RAGChatbot()
    analytics = ReceiptAnalytics()
    return ocr, rag, analytics

ocr_processor, rag_chatbot, receipt_analytics = init_components()

# Create necessary directories
os.makedirs("data/uploads", exist_ok=True)

# Title
st.title("📋 Receipt Verification System")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Upload Receipt", "Query Records", "Analytics Dashboard"])
    
    st.markdown("---")
    st.info("💡 **Tip**: Upload receipts to build your database, then query past records or view trends.")

# Page 1: Upload Receipt
if page == "Upload Receipt":
    st.header("📤 Upload Receipt")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Step 1: Upload Image")
        uploaded_file = st.file_uploader("Choose a receipt image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)
            
            if st.button("🔍 Process Receipt", type="primary"):
                with st.spinner("Processing receipt with AI..."):
                    # Save uploaded file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/uploads/receipt_{timestamp}.jpg"
                    
                    with open(filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Show progress
                    progress_placeholder = st.empty()
                    progress_placeholder.info("🔍 Analyzing receipt image...")
                    
                    # Process with OCR - now returns LIST of items
                    extracted_items = ocr_processor.process_receipt(filename)
                    
                    progress_placeholder.empty()
                    
                    if extracted_items and len(extracted_items) > 0:
                        st.session_state.extracted_items = extracted_items
                        st.session_state.image_path = filename
                        
                        shop_name = extracted_items[0].get('shop_name', 'Unknown')
                        total = sum(item.get('total_price', 0) for item in extracted_items)
                        
                        st.success(f"✅ Receipt processed! Found {len(extracted_items)} item(s) from {shop_name} - Total: ${total:.2f}")
                    else:
                        st.warning("⚠️ Could not extract information. Please fill in manually.")
                        st.session_state.extracted_items = [ocr_processor._get_default_data()]
                        st.session_state.image_path = filename
    
    with col2:
        st.subheader("Step 2: Verify & Save")
        
        if 'extracted_items' in st.session_state:
            items_list = st.session_state.extracted_items
            
            st.info(f"📋 Found {len(items_list)} item(s) on this receipt")
            
            # Common receipt info (outside form)
            st.markdown("**Receipt Information:**")
            shop_name = st.text_input("Shop Name", value=items_list[0].get('shop_name', ''), key="shop_name_input")
            
            col_date, col_mode = st.columns(2)
            with col_date:
                date_purchase = st.date_input("Date of Purchase", value=pd.to_datetime(items_list[0].get('date', datetime.now())), key="date_input")
            with col_mode:
                mode = st.selectbox("Mode of Purchase", ['Cash', 'Credit Card', 'Debit Card', 'E-Wallet', 'Other'], 
                                   index=['Cash', 'Credit Card', 'Debit Card', 'E-Wallet', 'Other'].index(items_list[0].get('mode', 'Other')),
                                   key="mode_input")
            
            st.markdown("---")
            st.markdown("**Items on Receipt:**")
            st.markdown("*Edit any field if needed*")
            
            # Create editable dataframe
            items_df_data = []
            for idx, item_data in enumerate(items_list):
                items_df_data.append({
                    'Item Name': item_data.get('item', ''),
                    'Quantity': int(item_data.get('unit', 1)),
                    'Unit Price': float(item_data.get('unit_price', 0.0)),
                    'Total Price': float(item_data.get('total_price', 0.0))
                })
            
            # Use data editor for easy editing
            edited_df = st.data_editor(
                pd.DataFrame(items_df_data),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Item Name": st.column_config.TextColumn("Item Name", required=True, width="medium"),
                    "Quantity": st.column_config.NumberColumn("Qty", min_value=1, step=1, format="%d", width="small"),
                    "Unit Price": st.column_config.NumberColumn("Unit Price ($)", min_value=0.0, step=0.01, format="%.2f", width="small"),
                    "Total Price": st.column_config.NumberColumn("Total ($)", min_value=0.0, step=0.01, format="%.2f", width="small"),
                }
            )
            
            # Calculate and show grand total
            grand_total = edited_df['Total Price'].sum()
            st.markdown(f"### Grand Total: ${grand_total:.2f}")
            
            st.markdown("---")
            
            # Save and reset buttons
            col_save, col_reset = st.columns(2)
            
            with col_save:
                if st.button("💾 Save All Items", type="primary", use_container_width=True, key="save_button"):
                    # Create records from edited dataframe
                    all_records = []
                    
                    for idx, row in edited_df.iterrows():
                        record = {
                            'Shop Name': shop_name,
                            'Date of Purchase': date_purchase.strftime('%Y-%m-%d'),
                            'Item Purchased': row['Item Name'],
                            'Mode of Purchase': mode,
                            'Unit Purchased': int(row['Quantity']),
                            'Unit Price': float(row['Unit Price']),
                            'Total Price': float(row['Total Price']),
                            'Image Path': st.session_state.image_path
                        }
                        all_records.append(record)
                    
                    # Save to CSV
                    csv_path = "data/receipts.csv"
                    new_df = pd.DataFrame(all_records)
                    
                    if os.path.exists(csv_path):
                        existing_df = pd.read_csv(csv_path)
                        final_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        final_df = new_df
                    
                    final_df.to_csv(csv_path, index=False)
                    
                    # Add to RAG database
                    for record in all_records:
                        rag_chatbot.add_receipt(record)
                    
                    st.success(f"✅ Saved {len(all_records)} items successfully! Total: ${grand_total:.2f}")
                    
                    # Show saved items
                    with st.expander("📋 View Saved Items"):
                        display_df = new_df.drop(columns=['Image Path'], errors='ignore')
                        st.dataframe(display_df, use_container_width=True)
                    
                    st.balloons()
                    
                    # Clear session state
                    del st.session_state.extracted_items
                    del st.session_state.image_path
                    
                    st.info("✨ You can now upload another receipt or go to Query/Analytics pages")
            
            with col_reset:
                if st.button("🔄 Reset", use_container_width=True, key="reset_button"):
                    del st.session_state.extracted_items
                    del st.session_state.image_path
                    st.rerun()

# Page 2: Query Records
elif page == "Query Records":
    st.header("💬 Query Receipt Records")
    
    st.markdown("""
    Ask questions about your receipt history:
    - *"Show me all purchases from Walmart"*
    - *"What did I buy last month?"*
    - *"Total spending on office supplies?"*
    - *"Find receipts above $100"*
    """)
    
    # Load existing data
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        st.info(f"📊 Database contains {len(df)} receipts")
    else:
        st.warning("⚠️ No receipts in database yet. Please upload some receipts first.")
        df = None
    
    # Chat interface
    query = st.text_input("Ask a question about your receipts:", placeholder="e.g., What were my purchases in January?")
    
    if st.button("🔍 Search", type="primary"):
        if query and df is not None and len(df) > 0:
            with st.spinner("Searching..."):
                response = rag_chatbot.query(query, df)
                
                st.subheader("📝 Answer")
                st.write(response['answer'])
                
                if response.get('relevant_receipts') is not None and len(response['relevant_receipts']) > 0:
                    st.subheader("🧾 Relevant Receipts")
                    st.dataframe(response['relevant_receipts'].drop(columns=['Image Path'], errors='ignore'), use_container_width=True)
        elif not query:
            st.warning("Please enter a question.")
        else:
            st.warning("No receipts to search. Please upload receipts first.")
    
    # Show all records
    if df is not None and len(df) > 0:
        with st.expander("📋 View All Records"):
            st.dataframe(df.drop(columns=['Image Path'], errors='ignore'), use_container_width=True)

# Page 3: Analytics Dashboard
elif page == "Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Receipts", len(df))
        with col2:
            st.metric("Total Spent", f"${df['Total Price'].sum():.2f}")
        with col3:
            st.metric("Avg Transaction", f"${df['Total Price'].mean():.2f}")
        with col4:
            unique_shops = df['Shop Name'].nunique()
            st.metric("Unique Shops", unique_shops)
        
        st.markdown("---")
        
        # Visualizations
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("💰 Spending by Shop")
            fig_shop = receipt_analytics.spending_by_shop(df)
            st.plotly_chart(fig_shop, use_container_width=True)
            
            st.subheader("💳 Payment Mode Distribution")
            fig_mode = receipt_analytics.payment_mode_distribution(df)
            st.plotly_chart(fig_mode, use_container_width=True)
        
        with col_b:
            st.subheader("📈 Spending Trend Over Time")
            fig_trend = receipt_analytics.spending_trend(df)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.subheader("🏷️ Top 10 Items Purchased")
            fig_items = receipt_analytics.top_items(df)
            st.plotly_chart(fig_items, use_container_width=True)
        
        # Category analysis
        st.markdown("---")
        st.subheader("📦 Spending by Category")
        
        df_categorized = receipt_analytics.categorize_purchases(df)
        fig_category = receipt_analytics.spending_by_category(df_categorized)
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Export options
        st.markdown("---")
        col_export1, col_export2 = st.columns([1, 3])
        
        with col_export1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"receipts_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ No receipts in database yet. Please upload some receipts first.")
        st.info("Upload receipts to see analytics and trends.")
