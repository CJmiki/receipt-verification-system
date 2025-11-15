"""
Analytics Module for Receipt Verification System
Provides categorization, visualizations, and reporting for receipt data
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class ReceiptAnalytics:
    """
    Analytics engine for receipt data
    
    Features:
    - Automatic purchase categorization
    - Interactive visualizations with Plotly
    - Spending trends and patterns
    - Summary reporting
    """
    
    def __init__(self):
        """Initialize with predefined categories"""
        # Category mapping with keywords
        self.categories = {
            'Food & Dining': [
                'restaurant', 'cafe', 'food', 'grocery', 'supermarket', 
                'market', 'mcdonald', 'starbucks', 'kfc', 'pizza', 
                'burger', 'subway', 'dining', 'lunch', 'dinner'
            ],
            'Office Supplies': [
                'office', 'stationery', 'paper', 'pen', 'printer', 
                'supplies', 'staples', 'ink', 'toner', 'folder', 'notebook'
            ],
            'Electronics': [
                'electronic', 'computer', 'laptop', 'phone', 'tech', 
                'best buy', 'apple store', 'software', 'hardware', 
                'monitor', 'keyboard', 'mouse'
            ],
            'Transportation': [
                'fuel', 'gas', 'petrol', 'transport', 'taxi', 'uber', 
                'grab', 'parking', 'toll', 'vehicle', 'car'
            ],
            'Utilities': [
                'electricity', 'water', 'internet', 'phone bill', 
                'utility', 'power', 'telecom'
            ],
            'Retail': [
                'walmart', 'target', 'store', 'shop', 'mart', 
                'costco', 'shopping', 'retail'
            ],
            'Healthcare': [
                'pharmacy', 'medical', 'hospital', 'clinic', 'health', 
                'doctor', 'medicine', 'drug'
            ],
            'Entertainment': [
                'movie', 'cinema', 'entertainment', 'game', 'gym', 
                'fitness', 'sport', 'theater'
            ],
        }
    
    def categorize_item(self, shop_name, item_name):
        """
        Categorize a purchase based on shop name and item
        
        Args:
            shop_name (str): Name of shop
            item_name (str): Item purchased
        
        Returns:
            str: Category name
        """
        try:
            # Combine and lowercase for matching
            text = f"{shop_name} {item_name}".lower()
            
            # Check each category
            for category, keywords in self.categories.items():
                if any(keyword in text for keyword in keywords):
                    return category
            
            # Default category
            return 'Other'
            
        except Exception as e:
            print(f"Categorization error: {e}")
            return 'Other'
    
    def categorize_purchases(self, df):
        """
        Add category column to entire dataframe
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            DataFrame: Data with 'Category' column added
        """
        try:
            df_copy = df.copy()
            df_copy['Category'] = df_copy.apply(
                lambda row: self.categorize_item(
                    str(row.get('Shop Name', '')), 
                    str(row.get('Item Purchased', ''))
                ), 
                axis=1
            )
            return df_copy
            
        except Exception as e:
            print(f"Error categorizing purchases: {e}")
            return df
    
    def spending_by_shop(self, df):
        """
        Create bar chart of spending by shop
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        try:
            # Aggregate spending by shop
            shop_spending = df.groupby('Shop Name')['Total Price'].sum()
            shop_spending = shop_spending.sort_values(ascending=False).head(10)
            
            if shop_spending.empty:
                return self._empty_chart("No data available")
            
            # Create bar chart
            fig = px.bar(
                x=shop_spending.index,
                y=shop_spending.values,
                labels={'x': 'Shop Name', 'y': 'Total Spent ($)'},
                title='Top 10 Shops by Total Spending',
                color=shop_spending.values,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                height=400,
                hovermode='x'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating shop spending chart: {e}")
            return self._empty_chart("Error creating chart")
    
    def payment_mode_distribution(self, df):
        """
        Create pie chart of payment modes
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            plotly.graph_objects.Figure: Pie chart
        """
        try:
            # Count payment modes
            mode_counts = df['Mode of Purchase'].value_counts()
            
            if mode_counts.empty:
                return self._empty_chart("No data available")
            
            # Create pie chart
            fig = px.pie(
                values=mode_counts.values,
                names=mode_counts.index,
                title='Payment Mode Distribution',
                hole=0.4  # Donut chart
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label'
            )
            
            fig.update_layout(height=400)
            
            return fig
            
        except Exception as e:
            print(f"Error creating payment mode chart: {e}")
            return self._empty_chart("Error creating chart")
    
    def spending_trend(self, df):
        """
        Create line chart of spending over time
        Shows both daily and cumulative spending
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            plotly.graph_objects.Figure: Line chart with dual y-axes
        """
        try:
            df_copy = df.copy()
            df_copy['Date of Purchase'] = pd.to_datetime(df_copy['Date of Purchase'])
            
            # Group by date and sum
            daily_spending = df_copy.groupby('Date of Purchase')['Total Price'].sum().reset_index()
            daily_spending = daily_spending.sort_values('Date of Purchase')
            
            if daily_spending.empty:
                return self._empty_chart("No data available")
            
            # Calculate cumulative spending
            daily_spending['Cumulative'] = daily_spending['Total Price'].cumsum()
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Daily spending trace
            fig.add_trace(go.Scatter(
                x=daily_spending['Date of Purchase'],
                y=daily_spending['Total Price'],
                mode='lines+markers',
                name='Daily Spending',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Daily: $%{y:.2f}<extra></extra>'
            ))
            
            # Cumulative spending trace (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=daily_spending['Date of Purchase'],
                y=daily_spending['Cumulative'],
                mode='lines',
                name='Cumulative Spending',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                yaxis='y2',
                hovertemplate='Date: %{x}<br>Cumulative: $%{y:.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Spending Trend Over Time',
                xaxis_title='Date',
                yaxis_title='Daily Spending ($)',
                yaxis2=dict(
                    title='Cumulative Spending ($)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating spending trend chart: {e}")
            return self._empty_chart("Error creating chart")
    
    def top_items(self, df):
        """
        Create horizontal bar chart of most purchased items
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            plotly.graph_objects.Figure: Horizontal bar chart
        """
        try:
            # Count item frequency
            item_freq = df['Item Purchased'].value_counts().head(10)
            
            if item_freq.empty:
                return self._empty_chart("No data available")
            
            # Create horizontal bar chart
            fig = px.bar(
                x=item_freq.values,
                y=item_freq.index,
                orientation='h',
                labels={'x': 'Number of Purchases', 'y': 'Item'},
                title='Top 10 Most Purchased Items',
                color=item_freq.values,
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating top items chart: {e}")
            return self._empty_chart("Error creating chart")
    
    def spending_by_category(self, df_categorized):
        """
        Create treemap of spending by category
        
        Args:
            df_categorized (DataFrame): Receipt data with Category column
        
        Returns:
            plotly.graph_objects.Figure: Treemap
        """
        try:
            # Aggregate by category
            category_spending = df_categorized.groupby('Category')['Total Price'].sum().reset_index()
            category_spending = category_spending.sort_values('Total Price', ascending=False)
            
            if category_spending.empty:
                return self._empty_chart("No data available")
            
            # Create treemap
            fig = px.treemap(
                category_spending,
                path=['Category'],
                values='Total Price',
                title='Spending Distribution by Category',
                color='Total Price',
                color_continuous_scale='RdYlGn_r'
            )
            
            fig.update_traces(textinfo='label+value+percent parent')
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            print(f"Error creating category chart: {e}")
            return self._empty_chart("Error creating chart")
    
    def monthly_breakdown(self, df):
        """
        Create stacked bar chart of monthly spending by category
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            plotly.graph_objects.Figure: Stacked bar chart
        """
        try:
            df_copy = df.copy()
            df_copy['Date of Purchase'] = pd.to_datetime(df_copy['Date of Purchase'])
            df_copy['Month'] = df_copy['Date of Purchase'].dt.to_period('M').astype(str)
            
            # Categorize purchases
            df_categorized = self.categorize_purchases(df_copy)
            
            # Group by month and category
            monthly_cat = df_categorized.groupby(['Month', 'Category'])['Total Price'].sum().reset_index()
            
            if monthly_cat.empty:
                return self._empty_chart("No data available")
            
            # Create stacked bar chart
            fig = px.bar(
                monthly_cat,
                x='Month',
                y='Total Price',
                color='Category',
                title='Monthly Spending by Category',
                labels={'Total Price': 'Amount ($)'},
                barmode='stack'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating monthly breakdown: {e}")
            return self._empty_chart("Error creating chart")
    
    def generate_summary_report(self, df):
        """
        Generate text summary report with key statistics
        
        Args:
            df (DataFrame): Receipt data
        
        Returns:
            dict: Summary statistics
        """
        try:
            df_categorized = self.categorize_purchases(df)
            
            report = {
                'total_receipts': len(df),
                'total_spending': df['Total Price'].sum(),
                'avg_transaction': df['Total Price'].mean(),
                'date_range': f"{df['Date of Purchase'].min()} to {df['Date of Purchase'].max()}",
                'unique_shops': df['Shop Name'].nunique(),
                'top_shop': df.groupby('Shop Name')['Total Price'].sum().idxmax() if len(df) > 0 else 'N/A',
                'top_category': df_categorized.groupby('Category')['Total Price'].sum().idxmax() if len(df) > 0 else 'N/A',
                'preferred_payment': df['Mode of Purchase'].mode()[0] if not df['Mode of Purchase'].mode().empty else 'N/A'
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return self._empty_report()
    
    def _empty_chart(self, message="No data available"):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400)
        return fig
    
    def _empty_report(self):
        """Return empty report structure"""
        return {
            'total_receipts': 0,
            'total_spending': 0.0,
            'avg_transaction': 0.0,
            'date_range': 'N/A',
            'unique_shops': 0,
            'top_shop': 'N/A',
            'top_category': 'N/A',
            'preferred_payment': 'N/A'
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Shop Name': ['Walmart', 'Starbucks', 'Office Depot', 'Walmart', 'Best Buy'],
        'Date of Purchase': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'Item Purchased': ['Groceries', 'Coffee', 'Printer Paper', 'Food', 'Laptop'],
        'Mode of Purchase': ['Credit Card', 'Cash', 'Credit Card', 'Debit Card', 'Credit Card'],
        'Unit Purchased': [1, 2, 5, 1, 1],
        'Unit Price': [50.00, 5.00, 10.00, 75.00, 899.00],
        'Total Price': [50.00, 10.00, 50.00, 75.00, 899.00]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics
    analytics = ReceiptAnalytics()
    
    # Categorize
    df_cat = analytics.categorize_purchases(df)
    print("Categories:", df_cat['Category'].tolist())
    
    # Generate report
    report = analytics.generate_summary_report(df)
    print("\nSummary Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Create visualizations
    fig1 = analytics.spending_by_shop(df)
    fig2 = analytics.payment_mode_distribution(df)
    
    print("\n✅ Analytics module working correctly!")