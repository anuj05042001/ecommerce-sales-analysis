import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stMetric"] {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e5e5;
}

[data-testid="stMetricValue"] {
    font-size: 25px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ecommerce_data.csv")

    # Date columns
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    df["ship_date"] = pd.to_datetime(
        df["ship_date"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "sales",
        "quantity",
        "discount",
        "profit",
        "shipping_cost"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Year
    df["Year"] = df["order_date"].dt.year

    return df


df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("🛒 E-Commerce Sales Analytics Dashboard")

st.markdown(
    "Interactive analysis of sales, profit, customers, products and regional performance."
)

st.divider()

# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header("🔎 Filters")

# Year
years = sorted(
    df["Year"].dropna().unique()
)

selected_years = st.sidebar.multiselect(
    "Year",
    years,
    default=years
)

# Category
categories = sorted(
    df["category"].dropna().unique()
)

selected_categories = st.sidebar.multiselect(
    "Category",
    categories,
    default=categories
)

# Region
regions = sorted(
    df["region"].dropna().unique()
)

selected_regions = st.sidebar.multiselect(
    "Region",
    regions,
    default=regions
)

# Segment
segments = sorted(
    df["segment"].dropna().unique()
)

selected_segments = st.sidebar.multiselect(
    "Segment",
    segments,
    default=segments
)

# ==================================================
# APPLY FILTERS
# ==================================================

filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["category"].isin(selected_categories)) &
    (df["region"].isin(selected_regions)) &
    (df["segment"].isin(selected_segments))
].copy()

# ==================================================
# EMPTY DATA CHECK
# ==================================================

if filtered_df.empty:

    st.warning(
        "No data available for the selected filters."
    )

    st.stop()

# ==================================================
# KPI CALCULATIONS
# ==================================================

total_sales = filtered_df["sales"].sum()

total_profit = filtered_df["profit"].sum()

total_quantity = filtered_df["quantity"].sum()

total_orders = filtered_df["order_id"].nunique()

profit_margin = (
    total_profit / total_sales * 100
    if total_sales != 0
    else 0
)

# ==================================================
# KPI CARDS
# ==================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "💰 Total Sales",
    f"${total_sales:,.0f}"
)

col2.metric(
    "📈 Total Profit",
    f"${total_profit:,.0f}"
)

col3.metric(
    "📦 Quantity Sold",
    f"{total_quantity:,.0f}"
)

col4.metric(
    "🧾 Total Orders",
    f"{total_orders:,}"
)

col5.metric(
    "📊 Profit Margin",
    f"{profit_margin:.2f}%"
)

st.divider()

# ==================================================
# SALES & PROFIT TREND
# ==================================================

st.subheader("📈 Sales & Profit Trend")

yearly = (
    filtered_df
    .groupby("Year", as_index=False)
    .agg(
        Sales=("sales", "sum"),
        Profit=("profit", "sum")
    )
)

fig = px.line(
    yearly,
    x="Year",
    y=["Sales", "Profit"],
    markers=True,
    title="Year-wise Sales and Profit"
)

fig.update_layout(
    hovermode="x unified",
    xaxis_title="Year",
    yaxis_title="Amount"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# CATEGORY ANALYSIS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📦 Sales by Category")

    category_sales = (
        filtered_df
        .groupby("category", as_index=False)["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
    )

    fig_category = px.bar(
        category_sales,
        x="category",
        y="sales",
        title="Category-wise Sales",
        text_auto=".2s"
    )

    fig_category.update_layout(
        xaxis_title="Category",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

with col2:

    st.subheader("💹 Profit by Category")

    category_profit = (
        filtered_df
        .groupby("category", as_index=False)["profit"]
        .sum()
        .sort_values(
            "profit",
            ascending=False
        )
    )

    fig_profit = px.bar(
        category_profit,
        x="category",
        y="profit",
        title="Category-wise Profit",
        text_auto=".2s"
    )

    fig_profit.update_layout(
        xaxis_title="Category",
        yaxis_title="Profit"
    )

    st.plotly_chart(
        fig_profit,
        use_container_width=True
    )

# ==================================================
# REGION ANALYSIS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🌍 Regional Sales")

    region_sales = (
        filtered_df
        .groupby("region", as_index=False)["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
    )

    fig_region = px.bar(
        region_sales,
        x="region",
        y="sales",
        title="Sales by Region",
        text_auto=".2s"
    )

    fig_region.update_layout(
        xaxis_title="Region",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_region,
        use_container_width=True
    )

with col2:

    st.subheader("🌍 Regional Profit")

    region_profit = (
        filtered_df
        .groupby("region", as_index=False)["profit"]
        .sum()
        .sort_values(
            "profit",
            ascending=False
        )
    )

    fig_region_profit = px.bar(
        region_profit,
        x="region",
        y="profit",
        title="Profit by Region",
        text_auto=".2s"
    )

    fig_region_profit.update_layout(
        xaxis_title="Region",
        yaxis_title="Profit"
    )

    st.plotly_chart(
        fig_region_profit,
        use_container_width=True
    )

# ==================================================
# CUSTOMER SEGMENT
# ==================================================

st.subheader("👥 Customer Segment Analysis")

segment_data = (
    filtered_df
    .groupby("segment", as_index=False)
    .agg(
        Sales=("sales", "sum"),
        Profit=("profit", "sum"),
        Orders=("order_id", "nunique")
    )
)

fig_segment = px.bar(
    segment_data,
    x="segment",
    y=["Sales", "Profit"],
    barmode="group",
    title="Sales & Profit by Customer Segment",
    text_auto=".2s"
)

st.plotly_chart(
    fig_segment,
    use_container_width=True
)

# ==================================================
# TOP PRODUCTS
# ==================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🏆 Top 10 Products by Sales")

    top_products = (
        filtered_df
        .groupby(
            "product_name",
            as_index=False
        )["sales"]
        .sum()
        .sort_values(
            "sales",
            ascending=False
        )
        .head(10)
        .sort_values("sales")
    )

    fig_top = px.bar(
        top_products,
        x="sales",
        y="product_name",
        orientation="h",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True
    )

with col2:

    st.subheader("⚠️ Bottom 10 Products by Profit")

    bottom_products = (
        filtered_df
        .groupby(
            "product_name",
            as_index=False
        )["profit"]
        .sum()
        .sort_values("profit")
        .head(10)
    )

    fig_bottom = px.bar(
        bottom_products,
        x="profit",
        y="product_name",
        orientation="h",
        title="Bottom 10 Products by Profit"
    )

    st.plotly_chart(
        fig_bottom,
        use_container_width=True
    )

# ==================================================
# SUB-CATEGORY ANALYSIS
# ==================================================

st.subheader("📊 Sub-Category Performance")

subcategory = (
    filtered_df
    .groupby(
        "sub_category",
        as_index=False
    )
    .agg(
        Sales=("sales", "sum"),
        Profit=("profit", "sum")
    )
)

fig_subcategory = px.bar(
    subcategory,
    x="sub_category",
    y=["Sales", "Profit"],
    barmode="group",
    title="Sales & Profit by Sub-Category",
    text_auto=".2s"
)

st.plotly_chart(
    fig_subcategory,
    use_container_width=True
)

# ==================================================
# DISCOUNT VS PROFIT
# ==================================================

st.subheader("💸 Discount vs Profit")

discount_data = filtered_df[
    [
        "discount",
        "profit",
        "sales",
        "product_name",
        "category",
        "region"
    ]
].copy()

# Convert to numeric
discount_data["discount"] = pd.to_numeric(
    discount_data["discount"],
    errors="coerce"
)

discount_data["profit"] = pd.to_numeric(
    discount_data["profit"],
    errors="coerce"
)

discount_data["sales"] = pd.to_numeric(
    discount_data["sales"],
    errors="coerce"
)

# Remove missing values
discount_data = discount_data.dropna(
    subset=[
        "discount",
        "profit",
        "sales"
    ]
)

# Only positive sales for bubble size
discount_data = discount_data[
    discount_data["sales"] > 0
]

# Limit records
if len(discount_data) > 5000:

    discount_data = discount_data.sample(
        n=5000,
        random_state=42
    )

# Only create chart if data exists
if not discount_data.empty:

    fig_discount = px.scatter(
        discount_data,
        x="discount",
        y="profit",
        size="sales",
        hover_data=[
            "product_name",
            "category",
            "region"
        ],
        title="Relationship Between Discount and Profit"
    )

    fig_discount.update_layout(
        xaxis_title="Discount",
        yaxis_title="Profit"
    )

    st.plotly_chart(
        fig_discount,
        use_container_width=True
    )

else:

    st.info(
        "No valid data available for Discount vs Profit analysis."
    )

# ==================================================
# DATA TABLE
# ==================================================

st.subheader("📋 Filtered Data")

st.caption(
    f"Showing {len(filtered_df):,} records based on selected filters."
)

display_columns = [
    "order_id",
    "order_date",
    "customer_name",
    "category",
    "sub_category",
    "product_name",
    "sales",
    "quantity",
    "discount",
    "profit",
    "region",
    "segment"
]

display_columns = [
    col
    for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    height=400
)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
    "**E-Commerce Sales Analysis | "
    "Built with Python, Pandas, Plotly & Streamlit**"
)
