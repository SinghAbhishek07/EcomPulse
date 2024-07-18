import streamlit as st
import pandas as pd
import numpy as np
import boto3
from PIL import Image
import altair as alt
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="EcomPulse Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# It's a sidebar navigation
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

# Created radio buttons for Graphs page
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

# Keeping the same background style for all pages
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

# Initializing a session using Amazon S3
s3 = boto3.client('s3')

# Defining the bucket and file key
bucket_name = 'temusdata'
vendor_file_key = 'processed/vendor_data.parquet'
product_file_key = 'processed/product_data.parquet'
merged_file_key = 'processed/merged_vendor_product.parquet'


# This is a function to fetch a file from S3 and read it into a pandas DataFrame
def fetch_parquet_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read()
    file_obj = BytesIO(file_content)
    df = pd.read_parquet(file_obj)
    return df

# Here we fetch the dataframes
vendor_df = fetch_parquet_from_s3(bucket_name, vendor_file_key)
product_df = fetch_parquet_from_s3(bucket_name, product_file_key)
merged_df = fetch_parquet_from_s3(bucket_name, merged_file_key)

# Defining the color scheme
colors = {
    'primary': '#090950',
    'secondary': '#ff7f0e',
    'background': '#f0f0f0',
    'text': '#000000'
}

# Creating plot of vendor statistics
def plot_vendor_statistics(merged_df):
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

    # Formatting the columns to one decimal point
    vendor_stats["total_sales"] = vendor_stats["total_sales"].map('{:.1f}'.format)
    vendor_stats["avg_sales"] = vendor_stats["avg_sales"].map('{:.1f}'.format)

    # Defining the column names for better readability
    column_names = [
        "VendorName", "Total Sales", "Avg Sales", "Shipping Cost", 
        "Customer Review Score", "Number of Feedbacks", "Total In Stock", 
        "Total Low Stock", "Total Out of Stock", "Total Electronics", 
        "Total Accessories", "Total Office Supplies"
    ]

    # Creating a table to display the statistics
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


# Defining plot functions
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

# Creating plot of sale_price_distribution
def plot_sale_price_distribution(product_df):
    fig = px.violin(product_df, x="Category", y="Sale Price", box=True, points="all",
                    title="Category-wise Sale Price Distribution",
                    color_discrete_sequence=[colors['primary']])
    fig.update_layout(paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font=dict(color=colors['text']))
    return fig

# Creating plot of vendor_performance
def plot_vendor_performance(vendor_df):
    fig = px.scatter(vendor_df, x="Customer Review Score", y="Number of Feedbacks",
                     size="Shipping Cost", color="VendorName",
                     hover_name="VendorName", log_x=True, size_max=60,
                     title="Vendor Performance: Customer Review Score vs. Number of Feedbacks")
    fig.update_layout(paper_bgcolor=colors['background'], plot_bgcolor=colors['background'], font=dict(color=colors['text']))
    return fig


# Modifying the product dataFrame by coping it to another dataframe named product_data 
product_data = product_df.copy()
product_data['Item'] = product_data['Item'].str.replace(r'\d+', '', regex=True).str.strip()

# Creating plot of item_proportion
def plot_item_proportion(product_data):
    item_counts = product_data['Item'].value_counts().reset_index()
    item_counts.columns = ['Item', 'Count']
    fig = px.pie(item_counts, names='Item', values='Count', 
                 title='Proportion of Each Item')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
    return fig

# Creating plot of enhanced_price_vs_stock_by_category
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

# Creating plot of enhanced_price_vs_stock_by_vendor
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


# Creating a copy of the processed product dataFrame
encoded_product_df = product_df.copy()

# Initializing the LabelEncoder
label_encoder = LabelEncoder()

# Encoding categorical variables in the copied DataFrame
encoded_product_df['Category'] = label_encoder.fit_transform(encoded_product_df['Category'])
encoded_product_df['VendorName'] = label_encoder.fit_transform(encoded_product_df['VendorName'])
encoded_product_df['Stock Status'] = label_encoder.fit_transform(encoded_product_df['Stock Status'])

# Creating a copy of the processed vendor dataFrame
standardized_vendor_df = vendor_df.copy()

# Initializing the StandardScaler
scaler = StandardScaler()

# Standardize the numerical columns in the copied DataFrame
standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']] = scaler.fit_transform(
standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])
    
# Creating plot of elbow_method
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

# We assume that optimal clusters are 3 based on the Elbow method
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
standardized_vendor_df['Cluster'] = kmeans.fit_predict(standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])


# Creating plot for cluster analysis
def plot_vendor_clusters_3d(vendor_df):
  
    # Applying KMeans clustering
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    standardized_vendor_df['Cluster'] = kmeans.fit_predict(standardized_vendor_df[['Shipping Cost', 'Customer Review Score', 'Number of Feedbacks']])
    standardized_vendor_df['VendorName'] = vendor_df['VendorName']  # Add original vendor names

    fig = px.scatter_3d(
        standardized_vendor_df, 
        x='Shipping Cost', 
        y='Customer Review Score', 
        z='Number of Feedbacks', 
        color='Cluster', 
        hover_name='VendorName', 
        title='3D Scatter Plot of Vendor Clusters'  # Ensure the color list matches the number of clusters
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(coloraxis_colorbar=dict(title='Cluster'), 
                      paper_bgcolor='#cacfd2',  # Change paper background color to light gray
                      plot_bgcolor='#cacfd2',  # Change plot background color to light gray
                      font=dict(color='black'))
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
                with st.expander("Observations"):
                    st.write("""
                    - GadgetUniverse has the highest total sale price, while Innovatech has the lowest.
                    - TechSphere has the highest average sale price, indicating higher-priced items.
                    - There is a notable variation in average sale price among vendors, with some having a high total sale price but a lower average sale price, indicating high volume but lower price items.
                    """)
            elif selection == "Category Sale Distribution":
                st.plotly_chart(plot_sale_price_distribution(product_df), use_container_width=True)
                with st.expander("Observations"):
                    st.write("""
                    - **Electronics:**
                        - Electronics have the widest range of sale prices, with a median price of 707.33. The interquartile range shows a significant spread from 397.67 to 1033.98, indicating varied pricing within this category.
                    - **Accessories:**
                        - Accessories also have a wide price range, though slightly less than electronics. The median price is higher than electronics at 746.36. The prices are more evenly distributed within the interquartile range of 403.44 to 1070.15.
                    - **Office Supplies:**
                        - Office supplies have a similar price distribution to accessories. The median price is 715.64, and the interquartile range is from 338.8 to 1064. This category has fewer extreme values, indicating a more consistent pricing strategy.
                    """)
                with st.expander("Findings"):
                    st.write("""
                    - Electronics have the highest variability in prices, indicating a broad range of products with different pricing strategies.
                    - Accessories and office supplies have similar price distributions, with fewer outliers and a more concentrated price range around their medians.
                    - Overall, the median prices across categories are relatively close, but the range and distribution of prices vary significantly.
                    """)
            elif selection == "Vendor Review Performance":
                st.plotly_chart(plot_vendor_performance(vendor_df), use_container_width=True)
                with st.expander("Observations"):
                    st.write("""
                    - The x-axis represents the customer review score, and the y-axis represents the number of feedbacks.
                    - Each bubble represents a vendor, distinguished by unique colors.
                    - The size of the bubble indicates the shipping cost for that vendor.
                    - **ElectroBuddy and GadgetUniverse:**
                        - High customer review scores and a large number of feedbacks.
                        - Indicate strong performance and high customer satisfaction.
                    - **Innovatech:**
                        - Lowest customer review scores and the least number of feedbacks.
                        - Indicates areas for improvement in both product quality and customer engagement.
                    - This plot helps to identify which vendors are performing well in terms of customer satisfaction and feedback volume, and which ones need to improve. The shipping cost indicated by bubble size also adds an additional layer of insight into vendor performance.
                    """)
            elif selection == "Item Proportions":
                st.plotly_chart(plot_item_proportion(product_data), use_container_width=True)
                with st.expander("Observations"):
                    st.write("""
                    - The pie chart represents the proportion of each grouped item in the dataset after removing suffixes and grouping similar items.
                    - Each category is color-coded and labeled with its corresponding percentage of the total items.
                    """)
                with st.expander("Key Insights"):
                    st.write("""
                    - The distribution of item categories is relatively even, with each category representing approximately 9% of the total dataset.
                        - Smartphones and laptops have a slightly higher proportion at 9.5%, making them the most prevalent item category.
                        - This balanced distribution suggests a diverse range of products offered, with no single category dominating the dataset.
                        - The even representation of item categories indicates a well-rounded inventory without any significant bias towards any specific product type.
                     """)
            elif selection == "Category Stock Analysis":
                st.plotly_chart(plot_enhanced_price_vs_stock_by_category(product_df), use_container_width=True)
                with st.expander("Observations"):
                    st.write("""
                    - The plot shows the distribution of sale prices across different stock statuses for each category (Electronics, Accessories, Office Supplies).
                    - **Electronics:**
                        - The median sale price for 'Low Stock' electronics is higher than for 'Out of Stock' and 'In Stock' items.
                        - There is a significant price range, with prices for 'In Stock' items being the most varied.
                    - **Accessories:**
                        - The median sale prices for 'In Stock' accessories are slightly higher than for 'Low Stock' and 'Out of Stock' items.
                        - The price range for accessories is narrower compared to electronics.
                    - **Office Supplies:**
                        - The median sale price for 'In Stock' office supplies is also higher compared to 'Low Stock' and 'Out of Stock' items.
                        - The price distribution is more consistent, with fewer extreme values.
                    - Overall, 'In Stock' items tend to have higher median sale prices across all categories, indicating that availability may have a positive impact on pricing. The variability in prices is highest for electronics, followed by accessories and office supplies.
                    """)

            elif selection == "Vendor Stock Analysis":
                st.plotly_chart(plot_enhanced_price_vs_stock_by_vendor(merged_df), use_container_width=True)
                with st.expander("Observations"):
                    st.write("""
                    - The plot shows the distribution of sale prices across different stock statuses for each vendor. Key insights include:
                    - **ElectroWorld:**
                        - The median sale price for 'In Stock' items is higher than for 'Out of Stock' and 'Low Stock' items.
                        - Sale prices are relatively consistent across stock statuses, with 'In Stock' items showing slightly higher variability.
                    - **MantaElectronics:**
                        - 'In Stock' items have the highest median sale price, with 'Low Stock' items having a slightly lower median price.
                        - 'Out of Stock' items have a lower median price, with a narrower price range.
                    - **GadgetUniverse:**
                        - 'Low Stock' items have the highest median sale price, with 'In Stock' items having the lowest median price.
                        - The price range for 'In Stock' items is wide, indicating varied pricing.
                    - **TechGiant:**
                        - The sale prices are consistent across different stock statuses.
                        - 'In Stock' items have a higher median price compared to 'Out of Stock' and 'Low Stock' items.
                    - **TechSphere:**
                        - 'Low Stock' items have the highest median sale price, with 'In Stock' items having the lowest median price.
                        - 'In Stock' items have the lowest median price, with a narrower price range.
                    - **ElectroBuddy:**
                        - 'Low Stock' items have the highest median sale price, with 'In Stock' and 'Out of Stock' items having lower median prices.
                        - The price range for 'In Stock' items is wide, indicating varied pricing.
                    - **Innovatech:**
                        - 'In Stock' items have the highest median sale price, with 'Out of Stock' items having the lowest median price.
                        - The price range for 'In Stock' items is wide, indicating varied pricing.
                    - Overall, 'In Stock' and 'Low Stock' items tend to have higher median sale prices across all vendors, indicating that availability positively impacts pricing. The variability in prices is noticeable for vendors like GadgetUniverse, TechSphere, and Innovatech, while others like ElectroWorld and TechGiant have more consistent pricing across stock statuses.
                    """)
            elif selection == "Cluster Insights":
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_elbow_method(vendor_df), use_container_width=True)
                with col2:
                    st.plotly_chart(plot_vendor_clusters_3d(vendor_df), use_container_width=True)
                with st.expander(" Elbow method Observation"):
                    st.write("""
                        - The elbow method helps determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters.
                        - The optimal number of clusters is identified at the "elbow point," where the WCSS starts to diminish at a slower rate which is 3 in our case
                    """)
                with st.expander("3D Scatter Plot of Vendor Clusters Observations"):
                    st.write("""
                        - The plot visualizes the clusters in a 3D space, with each point representing a vendor.
                        - **Axes:**
                            - X-axis: Shipping Cost (standardized)
                            - Y-axis: Customer Review Score (standardized)
                            - Z-axis: Number of Feedbacks (standardized)
                            - Color: Indicates the cluster to which each vendor belongs.
                    """)
                with st.expander("Findings"):
                    st.write("""
                    - **Cluster 0:**
                        - Vendors like ElectroBuddy, ElectroWorld, and TechSphere have higher customer review scores and a moderate number of feedbacks, associated with lower shipping costs.
                    - **Cluster 1:**
                        - Vendors like TechGiant, GadgetUniverse, and MantaElectronics have lower customer review scores but higher numbers of feedbacks.
                    - **Cluster 2:**
                        - Innovatech stands out with the highest shipping costs and the lowest number of feedbacks, indicating potential inefficiencies and lower customer satisfaction.
                    - This analysis provides a clear segmentation of vendors based on performance metrics, aiding in targeted strategies for improvement and optimization.
                    """)

        # Displaying the selected plot
        display_selected_plot(plot_selection)


# Data and Architecture page
elif page == "Architecture & Data":
    st.title("EcomPulse Architecture")

    try:
        img = Image.open("figs/Architecture-diagram.png")
        st.image(img, caption="EcomPulse Dashboard Architecture", use_column_width=True)
    except FileNotFoundError:
        st.error("Architecture image not found. Please check the path.")

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

    # Displaying the dataframes in a 2x3 grid
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