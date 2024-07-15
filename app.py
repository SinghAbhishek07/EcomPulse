


import streamlit as st
import pandas as pd
import numpy as np
import boto3
import altair as alt
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EcomPulse Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")



# Sidebar navigation
st.sidebar.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .sidebar-header {
        font-size: 20px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-title">EcoPulse</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-header">Project Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Go to", ["Home", "Graphs", "Data and Architecture"])

# Background style for all pages
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initialize a session using Amazon S3
s3 = boto3.client('s3')
# Define the bucket and file key

bucket_name = 'temusdata'
vendor_file_key = 'processed/vendor_data.parquet'
product_file_key = 'processed/product_data.parquet'
merged_file_key = 'processed/merged_vendor_product.parquet'


# Function to fetch a file from S3 and read it into a pandas DataFrame
def fetch_parquet_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read()
    file_obj = BytesIO(file_content)
    df = pd.read_parquet(file_obj)
    return df

# Fetch the dataframes
vendor_df = fetch_parquet_from_s3(bucket_name, vendor_file_key)
product_df = fetch_parquet_from_s3(bucket_name, product_file_key)
merged_df = fetch_parquet_from_s3(bucket_name, merged_file_key)

# Define color scheme
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'background': '#f0f0f0',
    'text': '#000000'
}
# Create vendor statistics plot
def plot_vendor_statistics(merged_df):
    # Calculate necessary metrics
    vendor_stats = merged_df.groupby("VendorName").agg(
        total_sales=pd.NamedAgg(column="Sale Price", aggfunc="sum"),
        avg_sales=pd.NamedAgg(column="Sale Price", aggfunc="mean"),
        shipping_cost=pd.NamedAgg(column="Shipping Cost", aggfunc="mean"),
        customer_review_score=pd.NamedAgg(column="Customer Review Score", aggfunc="mean"),
        number_of_feedbacks=pd.NamedAgg(column="Number of Feedbacks", aggfunc="mean"),
        total_in_stock=pd.NamedAgg(column="Stock Status", aggfunc=lambda x: (x == "In Stock").sum()),
        total_low_stock=pd.NamedAgg(column="Stock Status", aggfunc=lambda x: (x == "Low Stock").sum()),
        total_out_of_stock=pd.NamedAgg(column="Stock Status", aggfunc=lambda x: (x == "Out of Stock").sum()),
        total_electronics=pd.NamedAgg(column="Category", aggfunc=lambda x: (x == "Electronics").sum()),
        total_accessories=pd.NamedAgg(column="Category", aggfunc=lambda x: (x == "Accessories").sum()),
        total_office_supplies=pd.NamedAgg(column="Category", aggfunc=lambda x: (x == "Office Supplies").sum())
    ).reset_index()

    # Format the columns to one decimal point
    vendor_stats["total_sales"] = vendor_stats["total_sales"].map('{:.1f}'.format)
    vendor_stats["avg_sales"] = vendor_stats["avg_sales"].map('{:.1f}'.format)

    # Define the column names for better readability
    column_names = [
        "VendorName", "Total Sales", "Avg Sales", "Shipping Cost", 
        "Customer Review Score", "Number of Feedbacks", "Total In Stock", 
        "Total Low Stock", "Total Out of Stock", "Total Electronics", 
        "Total Accessories", "Total Office Supplies"
    ]

    # Create a table to display the statistics
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=column_names,
            fill_color=colors['primary'],
            align='left',
            font=dict(color=colors['background'], size=12)
        ),
        cells=dict(
            values=[vendor_stats[col] for col in vendor_stats.columns],
            fill_color=colors['background'],
            align='left',
            font=dict(color=colors['text'], size=10),
            height=30
        )
    )])

    fig.update_layout(
        title="Vendor Statistics",
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )
    return fig

# Define plot functions
def plot_total_and_avg_sale_price(merged_df):
    vendor_sales = merged_df.groupby("VendorName").agg(
        total_sale_price=pd.NamedAgg(column="Sale Price", aggfunc="sum"),
        avg_sale_price=pd.NamedAgg(column="Sale Price", aggfunc="mean")
    ).reset_index()

    bar = go.Bar(
        x=vendor_sales["VendorName"],
        y=vendor_sales["total_sale_price"],
        name="Total Sale Price",
        marker_color=colors['primary']
    )

    line = go.Scatter(
        x=vendor_sales["VendorName"],
        y=vendor_sales["avg_sale_price"],
        name="Average Sale Price",
        yaxis='y2',
        mode='lines+markers',
        marker=dict(color=colors['secondary'])
    )

    layout = go.Layout(
        title="Total and Average Sale Price by Vendor",
        xaxis=dict(title="VendorName"),
        yaxis=dict(title="Total Sale Price", titlefont=dict(color=colors['primary'])),
        yaxis2=dict(title="Average Sale Price", titlefont=dict(color=colors['secondary']), overlaying='y', side='right'),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )

    fig = go.Figure(data=[bar, line], layout=layout)
    return fig

def plot_sale_price_distribution(product_df):
    fig = px.violin(product_df, x="Category", y="Sale Price", box=True, points="all",
                    title="Category-wise Sale Price Distribution",
                    color_discrete_sequence=[colors['primary']])
    fig.update_layout(paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font=dict(color=colors['text']))
    return fig

def plot_vendor_performance(vendor_df):
    fig = px.scatter(vendor_df, x="Customer Review Score", y="Number of Feedbacks",
                     size="Shipping Cost", color="VendorName",
                     hover_name="VendorName", log_x=True, size_max=60,
                     title="Vendor Performance: Customer Review Score vs. Number of Feedbacks")
    fig.update_layout(paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font=dict(color=colors['text']))
    return fig

def plot_item_proportion(product_df):
    item_counts = product_df['Category'].value_counts().reset_index()
    item_counts.columns = ['Category', 'Count']

    fig = px.pie(item_counts, names='Category', values='Count', 
                 title='Proportion of Each Item')
    fig.update_layout(paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font=dict(color=colors['text']))
    return fig



st.markdown(
    """
    <style>
    .tight-layout {
        margin-top: -20px; /* Adjust this value to reduce gap between main plot and columns */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Home page
if page == "Home":
    st.title("Dashboard")
    st.plotly_chart(plot_vendor_statistics(merged_df))
   
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_total_and_avg_sale_price(merged_df))
    with col2:
        st.plotly_chart(plot_sale_price_distribution(product_df))

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_vendor_performance(vendor_df))
    with col4:
        st.plotly_chart(plot_item_proportion(product_df))

# Graphs page
elif page == "Graphs":
    st.title("Detailed Graphs")
   # st.plotly_chart(fig1)
  #  st.plotly_chart(fig2)
    # Add more detailed graphs if necessary

# Data and Architecture page
elif page == "Data and Architecture":
    st.title("Data and Architecture")
    st.header("Parquet Data")
    st.dataframe(product_df)
    st.dataframe(vendor_df)
    st.dataframe(merged_df)
    st.header("Project Architecture")
    st.image("path_to_architecture_image.jpg")