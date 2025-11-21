import streamlit as st
import pandas as pd
import os
import hashlib
from datetime import datetime
from ocr_processor import OCRProcessor  # Tesseract OCR - FREE!
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
os.makedirs("data/hashes", exist_ok=True)

# ==================== AUTHENTICATION ====================

def check_password():
    """Returns `True` if user is authenticated."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        # Simple authentication (in production, use proper auth)
        users = {
            "admin": {"password": "admin123", "role": "admin"},
            "user": {"password": "user123", "role": "user"}
        }
        
        if username in users and password == users[username]["password"]:
            st.session_state["authenticated"] = True
            st.session_state["role"] = users[username]["role"]
            st.session_state["username_logged"] = username
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["authenticated"] = False
    
    # First run, show login screen
    if "authenticated" not in st.session_state:
        st.title("🔐 Receipt Verification System")
        st.markdown("### Please Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.text_input("Username", key="username", placeholder="admin or user")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, type="primary", use_container_width=True)
            
            with st.expander("ℹ️ Demo Credentials"):
                st.markdown("""
                **Admin Account:**
                - Username: `admin`
                - Password: `admin123`
                - Access: Upload, Query, Analytics, Delete
                
                **User Account:**
                - Username: `user`
                - Password: `user123`
                - Access: Upload, Query only
                """)
        
        return False
    
    # Authentication failed
    if not st.session_state.get("authenticated", False):
        st.error("😕 Username or password incorrect")
        st.button("Try again", on_click=lambda: st.session_state.clear())
        return False
    
    # Authentication successful
    return True

if not check_password():
    st.stop()

# ==================== HELPER FUNCTIONS ====================

def get_file_hash(file_bytes):
    """Generate SHA256 hash of file content"""
    return hashlib.sha256(file_bytes).hexdigest()

def is_duplicate_receipt(file_hash):
    """Check if receipt was already uploaded"""
    hash_file = "data/hashes/uploaded_hashes.txt"
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            existing_hashes = f.read().splitlines()
            return file_hash in existing_hashes
    return False

def save_receipt_hash(file_hash):
    """Save receipt hash to prevent duplicates"""
    hash_file = "data/hashes/uploaded_hashes.txt"
    
    with open(hash_file, 'a') as f:
        f.write(f"{file_hash}\n")

def get_unique_receipts_count():
    """Get count of unique receipt uploads (not rows)"""
    hash_file = "data/hashes/uploaded_hashes.txt"
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return len(f.read().splitlines())
    return 0

def delete_receipt_by_image(image_path):
    """Delete all rows associated with a receipt image"""
    csv_path = "data/receipts.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Remove rows with this image path
        df = df[df['Image Path'] != image_path]
        df.to_csv(csv_path, index=False)
        return True
    return False

# ==================== HEADER ====================

# Show role and logout
col_header1, col_header2, col_header3 = st.columns([2, 1, 1])

with col_header1:
    st.title("📋 Receipt Verification System")

with col_header2:
    role_emoji = "👑" if st.session_state["role"] == "admin" else "👤"
    st.markdown(f"### {role_emoji} {st.session_state['username_logged'].title()}")
    st.caption(f"Role: {st.session_state['role'].title()}")

with col_header3:
    st.write("")  # Spacing
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("Navigation")
    
    # Role-based navigation
    if st.session_state["role"] == "admin":
        page = st.radio("Go to", ["Upload Receipt", "Query Records", "View All Records", "Analytics Dashboard", "Manage Receipts"])
    else:
        page = st.radio("Go to", ["Upload Receipt", "Query Records"])
    
    st.markdown("---")
    
    # Show stats
    unique_receipts = get_unique_receipts_count()
    st.metric("Unique Receipts", unique_receipts)
    
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        st.metric("Total Items", len(df))
        st.metric("Total Spent", f"${df['Total Price'].sum():.2f}")
    
    st.markdown("---")
    st.info(f"💡 **Tip**: {'As admin, you can view analytics and manage all receipts.' if st.session_state['role'] == 'admin' else 'Upload receipts and query your data.'}")

# ==================== PAGE 1: UPLOAD RECEIPT ====================

if page == "Upload Receipt":
    st.header("📤 Upload Receipt")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Step 1: Upload Image")
        uploaded_file = st.file_uploader("Choose a receipt image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Check for duplicate
            file_bytes = uploaded_file.getvalue()
            file_hash = get_file_hash(file_bytes)
            
            if is_duplicate_receipt(file_hash):
                st.error("❌ This receipt has already been uploaded!")
                st.warning("Duplicate receipts are not allowed. Please upload a different receipt.")
                st.stop()
            
            st.image(uploaded_file, caption="Uploaded Receipt", use_container_width=True)
            
            if st.button("🔍 Process Receipt", type="primary"):
                with st.spinner("Processing receipt with AI..."):
                    # Save uploaded file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/uploads/receipt_{timestamp}_{file_hash[:8]}.jpg"
                    
                    with open(filename, "wb") as f:
                        f.write(file_bytes)
                    
                    # Show progress
                    progress_placeholder = st.empty()
                    progress_placeholder.info("🔍 Analyzing receipt image...")
                    
                    # Process with OCR - returns LIST of items or None if not a receipt
                    extracted_items = ocr_processor.process_receipt(filename)
                    
                    progress_placeholder.empty()
                    
                    # Check if it's not a valid receipt
                    if extracted_items is None:
                        st.error("❌ This doesn't appear to be a receipt image")
                        st.warning("Please upload an image of a receipt, invoice, or bill. The image should contain:")
                        st.markdown("""
                        - Store or merchant name
                        - Purchased items with prices
                        - Payment information (total, tax, etc.)
                        - Transaction date
                        """)
                        st.info("💡 If this is a receipt but wasn't recognized, try taking a clearer photo with better lighting.")
                        st.stop()
                    
                    # Check if extraction was successful
                    if extracted_items and len(extracted_items) > 0:
                        first_item = extracted_items[0]
                        
                        # Check if data is actually extracted (not completely default)
                        # More lenient check - accept partial data
                        is_completely_empty = (
                            first_item.get('shop_name') == 'Unknown Store' and 
                            first_item.get('item') == 'Unknown Item' and 
                            first_item.get('total_price') == 0.0 and
                            len(extracted_items) == 1
                        )
                        
                        if is_completely_empty:
                            st.error("❌ Unable to extract receipt data")
                            st.warning("The image quality may be too poor or the receipt format is not recognized. Please try:")
                            st.markdown("""
                            - Taking a clearer photo with better lighting
                            - Ensuring the receipt is flat and all text is visible
                            - Using a different receipt image
                            - Manually entering the data below
                            """)
                            
                            # Provide manual entry option
                            st.session_state.extracted_items = [ocr_processor._get_default_data()]
                            st.session_state.image_path = filename
                            st.session_state.file_hash = file_hash
                        else:
                            # Accept partial extraction!
                            st.session_state.extracted_items = extracted_items
                            st.session_state.image_path = filename
                            st.session_state.file_hash = file_hash
                            
                            shop_name = extracted_items[0].get('shop_name', 'Unknown')
                            total = sum(item.get('total_price', 0) for item in extracted_items)
                            
                            # Check if extraction is partial
                            has_unknowns = (shop_name == 'Unknown Store' or 
                                          any(item.get('item') == 'Unknown Item' for item in extracted_items))
                            
                            if has_unknowns:
                                st.warning(f"⚠️ Partial extraction successful! Found {len(extracted_items)} item(s)")
                                st.info("💡 Some fields may be incomplete. Please verify and edit the data below before saving.")
                            else:
                                st.success(f"✅ Receipt processed! Found {len(extracted_items)} item(s) from {shop_name} - Total: ${total:.2f}")
                    else:
                        st.error("❌ Failed to process receipt")
                        st.warning("Please try again with a clearer image or manually enter the data.")
    
    with col2:
        st.subheader("Step 2: Verify & Save")
        
        if 'extracted_items' in st.session_state:
            items_list = st.session_state.extracted_items
            
            st.info(f"📋 Found {len(items_list)} item(s) on this receipt")
            
            # Common receipt info
            st.markdown("**Receipt Information:**")
            shop_name = st.text_input("Shop Name", value=items_list[0].get('shop_name', ''), key="shop_name_input")
            
            col_date, col_mode = st.columns(2)
            with col_date:
                date_purchase = st.date_input("Date of Purchase", value=pd.to_datetime(items_list[0].get('date', datetime.now())), key="date_input")
            with col_mode:
                # Feature 6: Editable payment mode with verification
                mode_options = ['Cash', 'Credit Card', 'Debit Card', 'E-Wallet', 'Other']
                current_mode = items_list[0].get('mode', 'Other')
                mode_index = mode_options.index(current_mode) if current_mode in mode_options else 4
                
                mode = st.selectbox(
                    "Mode of Purchase ✏️", 
                    mode_options,
                    index=mode_index,
                    key="mode_input",
                    help="Verify and edit the payment method if incorrect"
                )
            
            st.markdown("---")
            st.markdown("**Items on Receipt:**")
            st.caption("*Click any cell to edit. You can add/remove rows.*")
            
            # Create editable dataframe
            items_df_data = []
            for idx, item_data in enumerate(items_list):
                items_df_data.append({
                    'Item Name': item_data.get('item', ''),
                    'Quantity': float(item_data.get('unit', 1)),  # Feature 4: Allow decimals
                    'Unit Price': float(item_data.get('unit_price', 0.0)),  # Feature 5: Allow negative
                    'Total Price': float(item_data.get('total_price', 0.0))
                })
            
            # Feature 4 & 5: Decimal quantities and negative prices for discounts
            edited_df = st.data_editor(
                pd.DataFrame(items_df_data),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Item Name": st.column_config.TextColumn("Item Name", required=True, width="medium"),
                    "Quantity": st.column_config.NumberColumn(
                        "Qty (kg/units)", 
                        min_value=0.001, 
                        step=0.001, 
                        format="%.3f",  # 3 decimal places for weight
                        width="small",
                        help="Enter quantity (supports up to 3 decimal places for weights)"
                    ),
                    "Unit Price": st.column_config.NumberColumn(
                        "Unit Price ($)", 
                        step=0.01, 
                        format="%.2f", 
                        width="small",
                        help="Can be negative for discounts"
                    ),
                    "Total Price": st.column_config.NumberColumn(
                        "Total ($)", 
                        step=0.01, 
                        format="%.2f", 
                        width="small"
                    ),
                }
            )
            
            # Calculate and show grand total
            grand_total = edited_df['Total Price'].sum()
            
            col_total, col_items = st.columns(2)
            with col_total:
                st.metric("Grand Total", f"${grand_total:.2f}")
            with col_items:
                st.metric("Total Items", len(edited_df))
            
            st.markdown("---")
            
            # Save and reset buttons
            col_save, col_reset = st.columns(2)
            
            with col_save:
                if st.button("💾 Save All Items", type="primary", use_container_width=True, key="save_button"):
                    # Validate data
                    if shop_name.strip() == "":
                        st.error("❌ Shop name cannot be empty")
                    elif edited_df['Item Name'].str.strip().eq("").any():
                        st.error("❌ All items must have a name")
                    else:
                        # Create records from edited dataframe
                        all_records = []
                        
                        for idx, row in edited_df.iterrows():
                            record = {
                                'Shop Name': shop_name,
                                'Date of Purchase': date_purchase.strftime('%Y-%m-%d'),
                                'Item Purchased': row['Item Name'],
                                'Mode of Purchase': mode,
                                'Unit Purchased': round(float(row['Quantity']), 3),  # 3 decimals
                                'Unit Price': round(float(row['Unit Price']), 2),
                                'Total Price': round(float(row['Total Price']), 2),
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
                        
                        # Save receipt hash (Feature 7: Prevent duplicates)
                        save_receipt_hash(st.session_state.file_hash)
                        
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
                        del st.session_state.file_hash
                        
                        st.info("✨ You can now upload another receipt or go to Query page")
            
            with col_reset:
                if st.button("🔄 Reset", use_container_width=True, key="reset_button"):
                    del st.session_state.extracted_items
                    del st.session_state.image_path
                    if 'file_hash' in st.session_state:
                        del st.session_state.file_hash
                    st.rerun()

# ==================== PAGE 2: QUERY RECORDS ====================

elif page == "Query Records":
    st.header("💬 Query Receipt Records")
    
    st.markdown("""
    Ask questions about your receipt history:
    - *"Show me all purchases from NTUC"*
    - *"What did I buy last month?"*
    - *"Total spending on office supplies?"*
    - *"Find receipts above $100"*
    """)
    
    # Load existing data
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        
        # Feature 2: Show unique receipts count, not total rows
        unique_receipts = get_unique_receipts_count()
        total_items = len(df)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📊 Database contains **{unique_receipts} receipts** ({total_items} items)")
        with col_info2:
            st.metric("Total Spending", f"${df['Total Price'].sum():.2f}")
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

# ==================== PAGE 3: VIEW ALL RECORDS (ADMIN ONLY) ====================

elif page == "View All Records":
    if st.session_state["role"] != "admin":
        st.error("🚫 Access Denied: Admin only")
        st.stop()
    
    st.header("📋 All Receipt Records")
    
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        
        # Feature 2: Show correct counts
        unique_receipts = get_unique_receipts_count()
        total_items = len(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Receipts", unique_receipts)
        with col2:
            st.metric("Total Items", total_items)
        with col3:
            st.metric("Total Spent", f"${df['Total Price'].sum():.2f}")
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔍 Filters")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            shops = ['All'] + sorted(df['Shop Name'].unique().tolist())
            selected_shop = st.selectbox("Shop", shops)
        
        with col_f2:
            modes = ['All'] + sorted(df['Mode of Purchase'].unique().tolist())
            selected_mode = st.selectbox("Payment Mode", modes)
        
        with col_f3:
            date_filter = st.selectbox("Date Range", ['All', 'Last 7 days', 'Last 30 days', 'Last 90 days'])
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_shop != 'All':
            filtered_df = filtered_df[filtered_df['Shop Name'] == selected_shop]
        
        if selected_mode != 'All':
            filtered_df = filtered_df[filtered_df['Mode of Purchase'] == selected_mode]
        
        if date_filter != 'All':
            days = {'Last 7 days': 7, 'Last 30 days': 30, 'Last 90 days': 90}[date_filter]
            cutoff_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            filtered_df = filtered_df[filtered_df['Date of Purchase'] >= cutoff_date]
        
        st.markdown("---")
        st.subheader(f"📊 Results: {len(filtered_df)} items")
        
        # Display data
        display_df = filtered_df.drop(columns=['Image Path'], errors='ignore')
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"receipts_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No receipts in database yet.")

# ==================== PAGE 4: ANALYTICS DASHBOARD (ADMIN ONLY) ====================

elif page == "Analytics Dashboard":
    if st.session_state["role"] != "admin":
        st.error("🚫 Access Denied: Admin only")
        st.stop()
    
    st.header("📊 Analytics Dashboard")
    
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        
        # Feature 2: Show correct receipt count
        unique_receipts = get_unique_receipts_count()
        total_items = len(df)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Unique Receipts", unique_receipts)
        with col2:
            st.metric("Total Items", total_items)
        with col3:
            st.metric("Total Spent", f"${df['Total Price'].sum():.2f}")
        with col4:
            st.metric("Avg per Receipt", f"${df.groupby('Image Path')['Total Price'].sum().mean():.2f}")
        
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
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report",
            data=csv,
            file_name=f"receipts_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No receipts in database yet.")

# ==================== PAGE 5: MANAGE RECEIPTS (ADMIN ONLY) ====================

elif page == "Manage Receipts":
    if st.session_state["role"] != "admin":
        st.error("🚫 Access Denied: Admin only")
        st.stop()
    
    st.header("🗑️ Manage Receipts")
    st.warning("⚠️ **Warning:** Deleted receipts cannot be recovered!")
    
    if os.path.exists("data/receipts.csv"):
        df = pd.read_csv("data/receipts.csv")
        
        # Group by receipt (image path)
        receipts = df.groupby('Image Path').agg({
            'Shop Name': 'first',
            'Date of Purchase': 'first',
            'Mode of Purchase': 'first',
            'Total Price': 'sum',
            'Item Purchased': 'count'
        }).reset_index()
        
        receipts.columns = ['Image Path', 'Shop', 'Date', 'Payment', 'Total', 'Items']
        receipts = receipts.sort_values('Date', ascending=False)
        
        st.info(f"📊 Managing {len(receipts)} unique receipts")
        
        # Display receipts with delete option
        for idx, receipt in receipts.iterrows():
            with st.expander(f"🧾 {receipt['Shop']} - {receipt['Date']} - ${receipt['Total']:.2f} ({int(receipt['Items'])} items)"):
                col_info, col_action = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**Shop:** {receipt['Shop']}")
                    st.markdown(f"**Date:** {receipt['Date']}")
                    st.markdown(f"**Payment:** {receipt['Payment']}")
                    st.markdown(f"**Items:** {int(receipt['Items'])}")
                    st.markdown(f"**Total:** ${receipt['Total']:.2f}")
                    
                    # Show items
                    items_df = df[df['Image Path'] == receipt['Image Path']][['Item Purchased', 'Unit Purchased', 'Unit Price', 'Total Price']]
                    st.dataframe(items_df, use_container_width=True, hide_index=True)
                
                with col_action:
                    if st.button(f"🗑️ Delete", key=f"delete_{idx}", type="secondary"):
                        if delete_receipt_by_image(receipt['Image Path']):
                            st.success("✅ Receipt deleted!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete")
    else:
        st.info("No receipts to manage yet.")
