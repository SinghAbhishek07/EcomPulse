


import streamlit as st
import pandas as pd
import numpy as np
import boto3
import altair as alt
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from PIL import Image



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

st.sidebar.markdown('<div class="sidebar-title">EcomPulse</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-header">Project Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Go to", ["Home", "Graphs", "Architecture & Data"])

# Conditional radio buttons for Graphs page
plot_selection = None
if page == "Graphs":
    plot_selection = st.sidebar.radio(
        "Select a plot to display:",
        ("Vendor Sales Summary", 
         "Category Sale Distribution",
         "Vendor Review Performance",
         "Item Proportions",
         "Category Stock Analysis",
         "Vendor Stock Analysis",
         "Cluster Insights")
    )

# Background style for all pages
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E1E1E5;
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
    'primary': '#090950',
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
            font=dict(color=colors['background'], size=13)
        ),
        cells=dict(
            values=[vendor_stats[col] for col in vendor_stats.columns],
            fill_color=colors['background'],
            align='left',
            font=dict(color=colors['text'], size=12),
            height=40
        )
    )])

    fig.update_layout(
        title="Vendor Statistics",
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
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
# Modify DataFrame
product_data = product_df.copy()
product_data['Item'] = product_data['Item'].str.replace(r'\d+', '', regex=True).str.strip()



def plot_item_proportion(product_data):
    item_counts = product_data['Item'].value_counts().reset_index()
    item_counts.columns = ['Item', 'Count']
    fig = px.pie(item_counts, names='Item', values='Count', 
                 title='Proportion of Each Item')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
    return fig


def plot_enhanced_price_vs_stock_by_category(df):
    categories = df["Category"].unique()
    fig = make_subplots(rows=len(categories), cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, subplot_titles=categories)

    for i, category in enumerate(categories):
        category_df = df[df["Category"] == category]
        
        box_trace = go.Box(
            x=category_df["Stock Status"],
            y=category_df["Sale Price"],
            name=f"{category} Sale Price",
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1
        )
        fig.add_trace(box_trace, row=i+1, col=1)

        mean_sale_price = category_df.groupby("Stock Status")["Sale Price"].mean().reset_index()
        scatter_trace = go.Scatter(
            x=mean_sale_price["Stock Status"],
            y=mean_sale_price["Sale Price"],
            mode='markers+lines',
            name=f"{category} Average Price",
            marker=dict(size=10, color='red'),
            line=dict(dash='dash', color='red')
        )
        fig.add_trace(scatter_trace, row=i+1, col=1)

    fig.update_layout(height=600, width=800, title_text="Sale Price vs Stock Status by Category", showlegend=False)
    fig.update_yaxes(title_text="Sale Price")
    return fig

# Plot 2: Enhanced Price vs Stock by Vendor
def plot_enhanced_price_vs_stock_by_vendor(df):
    vendors = df["VendorName"].unique()
    fig = make_subplots(rows=len(vendors), cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, subplot_titles=vendors)

    for i, vendor in enumerate(vendors):
        vendor_df = df[df["VendorName"] == vendor]
        
        box_trace = go.Box(
            x=vendor_df["Stock Status"],
            y=vendor_df["Sale Price"],
            name=f"{vendor} Sale Price",
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1
        )
        fig.add_trace(box_trace, row=i+1, col=1)

        mean_sale_price = vendor_df.groupby("Stock Status")["Sale Price"].mean().reset_index()
        scatter_trace = go.Scatter(
            x=mean_sale_price["Stock Status"],
            y=mean_sale_price["Sale Price"],
            mode='markers',
            name=f"{vendor} Average Price",
            marker=dict(size=10, color='red')
        )
        fig.add_trace(scatter_trace, row=i+1, col=1)

    fig.update_layout(height=2000, width=1000, title_text="Sale Price vs Stock Status by Vendor", showlegend=False)
    fig.update_yaxes(title_text="Sale Price")
    return fig


# Create a copy of the original DataFrame
encoded_product_df = product_df.copy()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical variables in the copied DataFrame
encoded_product_df['Category'] = label_encoder.fit_transform(encoded_product_df['Category'])
encoded_product_df['VendorName'] = label_encoder.fit_transform(encoded_product_df['VendorName'])
encoded_product_df['Stock Status'] = label_encoder.fit_transform(encoded_product_df['Stock Status'])

standardized_vendor_df = vendor_df.copy()

    # Initialize the StandardScaler
scaler = StandardScaler()

    # Standardize the numerical columns in the copied DataFrame
standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']] = scaler.fit_transform(
standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])
    

def plot_elbow_method(vendor_df):
    

    wcss = []
    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])
        wcss.append(kmeans.inertia_)

    # Create the Elbow graph using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, 8)),
        y=wcss,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2),
        name='WCSS'
    ))

    fig.update_layout(
        title='Elbow Method',
        xaxis_title='Number of clusters',
        yaxis_title='WCSS',
        showlegend=False,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )
    return fig

# Assume optimal clusters are 3 based on the Elbow method
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
standardized_vendor_df['Cluster'] = kmeans.fit_predict(standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])

# Create a custom color sequence
custom_colors = ['#636EFA', '#EF553B', '#00CC96']

def plot_vendor_clusters_3d(vendor_df):
  
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    standardized_vendor_df['Cluster'] = kmeans.fit_predict(standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])
    standardized_vendor_df['VendorName'] = vendor_df['VendorName']  # Add original vendor names

    # Create a 3D scatter plot with Plotly
    fig = px.scatter_3d(
        standardized_vendor_df, 
        x='Shipping Cost', 
        y='Customer Review Score', 
        z='Number of Feedbacks', 
        color='Cluster', 
        hover_name='VendorName', 
        title='3D Scatter Plot of Vendor Clusters',
        color_discrete_sequence=custom_colors
    )

    # Update marker size and color bar label
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(coloraxis_colorbar=dict(title='Cluster'), 
                      paper_bgcolor=colors['background'], 
                      plot_bgcolor=colors['background'], 
                      font=dict(color=colors['text']))
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
        st.plotly_chart(plot_item_proportion(product_data))

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_enhanced_price_vs_stock_by_category(product_df))
    with col6:
        st.plotly_chart(plot_vendor_clusters_3d(vendor_df))

# Graphs page
elif page == "Graphs":
    if plot_selection:
        # Function to display the selected plot
        def display_selected_plot(selection):
            plot_titles = {
                "Vendor Sales Summary": "Total and Average Sale Price by Vendor",
                "Category Sale Distribution": "Category-wise Sale Price Distribution",
                "Vendor Review Performance": "Vendor Performance: Customer Review Score vs. Number of Feedbacks",
                "Item Proportions": "Proportion of Each Item",
                "Category Stock Analysis": "Sale Price vs Stock Status by Category",
                "Vendor Stock Analysis": "Sale Price vs Stock Status by Vendor",
                "Cluster Insights": "Cluster Analysis"
            }


            st.header(plot_titles[selection])

            if selection == "Vendor Sales Summary":
                st.plotly_chart(plot_total_and_avg_sale_price(merged_df), use_container_width=True)
            elif selection == "Category Sale Distribution":
                st.plotly_chart(plot_sale_price_distribution(product_df), use_container_width=True)
            elif selection == "Vendor Review Performance":
                st.plotly_chart(plot_vendor_performance(vendor_df), use_container_width=True)
            elif selection == "Item Proportions":
                st.plotly_chart(plot_item_proportion(product_data), use_container_width=True)
            elif selection == "Category Stock Analysis":
                st.plotly_chart(plot_enhanced_price_vs_stock_by_category(product_df), use_container_width=True)
            elif selection == "Vendor Stock Analysis":
                st.plotly_chart(plot_enhanced_price_vs_stock_by_vendor(merged_df), use_container_width=True)
            elif selection == "Cluster Insights":
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_elbow_method(vendor_df), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_vendor_clusters_3d(vendor_df), use_container_width=True)

        # Display the selected plot
        display_selected_plot(plot_selection)

        # Text box for observations and findings
        st.text_area("Observations and Findings:", height=200)


   # st.plotly_chart(fig1)
  #  st.plotly_chart(fig2)
    # Add more detailed graphs if necessary

# Data and Architecture page
elif page == "Architecture & Data":
    st.title("EcomPulse Architecture")

    try:
        img = Image.open("Architecture-diagram.png")
        st.image(img, caption="EcomPulse Dashboard Architecture", use_column_width=True)
    except FileNotFoundError:
        st.error("Architecture image not found. Please check the path.")

    # Display dataframes in 2x3 grid format
    st.header("Dataframes")

    dataframes = {
        "Vendor": vendor_df,
        "Product": product_df,
        "Merged": merged_df,
        "Product_Data": product_data,
        "Encoded_Product": encoded_product_df,
        "Standardized_Vendor": standardized_vendor_df
    }

    df_names = list(dataframes.keys())
    dfs = list(dataframes.values())

    # Display dataframes in a 2x3 grid
    rows = 2
    cols = 3
    for i in range(0, len(dfs), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(dfs):
                df_name = df_names[i + j]
                df = dfs[i + j]
                col.subheader(df_name)
                col.dataframe(df, height=200)