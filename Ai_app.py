import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient
import json
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"

NUMERIC_COLUMNS = [
    "sales",
    "quantity",
    "discount",
    "profit",
    "shipping_cost"
]

GROUP_COLUMNS = [
    "category",
    "sub_category",
    "region",
    "segment",
    "market",
    "country",
    "state",
    "ship_mode",
    "product_name",
    "customer_name"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 20px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    background-color: #fafafa;
    margin-bottom: 15px;
}

.small-note {
    color: #777;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🤖 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your e-commerce dataset in natural English.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("ecommerce_data.csv")

    # ----------------------------
    # Date conversion
    # ----------------------------

    if "order_date" in data.columns:

        data["order_date"] = pd.to_datetime(
            data["order_date"],
            errors="coerce"
        )

    if "ship_date" in data.columns:

        data["ship_date"] = pd.to_datetime(
            data["ship_date"],
            errors="coerce"
        )

    # ----------------------------
    # Numeric conversion
    # ----------------------------

    for column in NUMERIC_COLUMNS:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # ----------------------------
    # Year
    # ----------------------------

    if "order_date" in data.columns:

        data["Year"] = (
            data["order_date"]
            .dt.year
        )

    return data


df = load_data()


# ============================================================
# DATA INFORMATION
# ============================================================

with st.expander("📊 Dataset Information"):

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    c3.metric(
        "Total Sales",
        f"${df['sales'].sum():,.0f}"
        if "sales" in df.columns
        else "N/A"
    )

    c4.metric(
        "Total Profit",
        f"${df['profit'].sum():,.0f}"
        if "profit" in df.columns
        else "N/A"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 AI Data Analyst")

st.sidebar.caption(
    "Ask questions in normal English."
)

st.sidebar.divider()

st.sidebar.subheader("💡 Try asking")

questions = [

    "What is the total sales?",

    "Which category is most profitable?",

    "Which region has the highest sales?",

    "Show sales by category.",

    "Show profit by region.",

    "How did sales change over the years?",

    "What was the total profit in 2014?",

    "Show the top 10 products by sales.",

    "Show the worst 10 products by profit.",

    "Which products are losing money?",

    "What is the average discount?",

    "Does discount affect profit?",

    "Which segment generates the most sales?",

    "Give me 5 important business insights."

]

for q in questions:

    st.sidebar.write(
        "• " + q
    )


st.sidebar.divider()

st.sidebar.info(
    "Calculations are performed using Pandas "
    "on the actual dataset."
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:

    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "HF_TOKEN is missing. "
        "Add it in Streamlit → Settings → Secrets."
    )

    st.stop()


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# HELPER: CLEAN JSON
# ============================================================

def clean_json(text):

    if not text:

        return None

    text = text.strip()

    # Remove Qwen thinking blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove markdown
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[start:end + 1]

    try:

        return json.loads(text)

    except Exception:

        return None


# ============================================================
# HELPER: SAFE INT
# ============================================================

def safe_int(value, default=10):

    if value is None:

        return default

    try:

        value = int(value)

    except (
        ValueError,
        TypeError
    ):

        return default

    if value < 1:

        return default

    return min(value, 100)


# ============================================================
# HELPER: NORMALIZE ANALYSIS
# ============================================================

def normalize_analysis(result):

    if not isinstance(result, dict):

        result = {}

    analysis_type = result.get(
        "analysis_type",
        "single_value"
    )

    valid_types = [

        "single_value",

        "group_by",

        "top_n",

        "bottom_n",

        "trend",

        "comparison",

        "negative_profit",

        "discount_profit",

        "insights"

    ]

    if analysis_type not in valid_types:

        analysis_type = "single_value"


    group_column = result.get(
        "group_column"
    )

    if (
        group_column
        not in df.columns
    ):

        group_column = None


    value_column = result.get(
        "value_column"
    )

    if (
        value_column
        not in df.columns
    ):

        value_column = None


    aggregation = result.get(
        "aggregation",
        "sum"
    )

    if aggregation not in [
        "sum",
        "mean",
        "count",
        "min",
        "max"
    ]:

        aggregation = "sum"


    n = safe_int(
        result.get("n"),
        10
    )


    year = result.get(
        "filter_year"
    )

    if year in [
        None,
        "",
        "null",
        "None"
    ]:

        year = None

    else:

        try:

            year = int(year)

        except (
            ValueError,
            TypeError
        ):

            year = None


    return {

        "analysis_type": analysis_type,

        "group_column": group_column,

        "value_column": value_column,

        "aggregation": aggregation,

        "n": n,

        "filter_year": year
    }


# ============================================================
# AI INTENT FUNCTION
# ============================================================

def get_ai_intent(question):

    system_prompt = f"""
You are an expert Data Analyst.

You are NOT responsible for calculating numbers.

Your job is to understand the user's natural-language
question and convert it into an analysis instruction.

The actual calculations will be performed by Python/Pandas.

Dataset columns:

{json.dumps(list(df.columns), indent=2)}

Available numeric columns:

{json.dumps(
    [c for c in NUMERIC_COLUMNS if c in df.columns],
    indent=2
)}

Available grouping columns:

{json.dumps(
    [c for c in GROUP_COLUMNS if c in df.columns],
    indent=2
)}

Return ONLY valid JSON.

Schema:

{{
    "analysis_type": "single_value",
    "group_column": null,
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

Allowed analysis_type:

single_value
group_by
top_n
bottom_n
trend
comparison
negative_profit
discount_profit
insights

Allowed aggregation:

sum
mean
count
min
max

Interpretation rules:

SALES:
sales, revenue, revenue generated, money made from sales
=> sales

PROFIT:
profit, profitability, earnings
=> profit

QUANTITY:
units, quantity, items sold
=> quantity

DISCOUNT:
discount, discount percentage
=> discount

SHIPPING:
shipping cost, delivery cost
=> shipping_cost

ORDERS:
orders, number of orders
=> order_id with count

CATEGORY:
category, categories
=> category

SUB-CATEGORY:
sub-category, subcategory
=> sub_category

REGION:
region, regions
=> region

SEGMENT:
segment, customer segment
=> segment

PRODUCT:
product, products
=> product_name

YEAR:
If user specifies 2011, 2012, 2013 or 2014,
put that year in filter_year.

Examples:

Question:
Which category has the highest sales?

Return:

{{
    "analysis_type": "group_by",
    "group_column": "category",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

Question:
Show top 5 products by sales.

Return:

{{
    "analysis_type": "top_n",
    "group_column": "product_name",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 5,
    "filter_year": null
}}

Question:
How did sales change over the years?

Return:

{{
    "analysis_type": "trend",
    "group_column": "Year",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

Question:
Does discount affect profit?

Return:

{{
    "analysis_type": "discount_profit",
    "group_column": null,
    "value_column": "profit",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

Question:
Which products are losing money?

Return:

{{
    "analysis_type": "negative_profit",
    "group_column": "product_name",
    "value_column": "profit",
    "aggregation": "sum",
    "n": 20,
    "filter_year": null
}}

Question:
Give me 5 important business insights.

Return:

{{
    "analysis_type": "insights",
    "group_column": null,
    "value_column": null,
    "aggregation": "sum",
    "n": 5,
    "filter_year": null
}}

Never calculate values.

Never invent columns.

Return JSON only.
"""


    user_prompt = f"""
User question:

{question}

Return JSON only.
"""


    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            max_tokens=500,

            temperature=0.0
        )


        raw = (
            response
            .choices[0]
            .message.content
        )


        return clean_json(raw)


    except Exception:

        return None


# ============================================================
# APPLY YEAR FILTER
# ============================================================

def apply_filters(data, analysis):

    filtered = data.copy()

    year = analysis.get(
        "filter_year"
    )

    if (
        year is not None
        and "Year" in filtered.columns
    ):

        filtered = filtered[
            filtered["Year"] == year
        ]

    return filtered


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("💬 Ask Your Data")

question = st.text_input(

    "Write your question in English:",

    placeholder=(
        "Example: Which category is most profitable?"
    ),

    key="question"
)


analyze = st.button(
    "🔍 Analyze",
    type="primary"
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    with st.spinner(
        "🤖 Understanding your question..."
    ):

        analysis = get_ai_intent(
            question
        )


    # --------------------------------------------------------
    # AI FAILURE
    # --------------------------------------------------------

    if analysis is None:

        st.warning(
            "I couldn't understand that question clearly."
        )

        st.info(
            "Try asking something like: "
            "'Which category has the highest sales?' "
            "or 'Show sales by region.'"
        )

        st.stop()


    analysis = normalize_analysis(
        analysis
    )


    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    analysis_df = apply_filters(
        df,
        analysis
    )


    if analysis_df.empty:

        st.warning(
            "No data was found for the selected filter."
        )

        st.stop()


    analysis_type = analysis[
        "analysis_type"
    ]

    group_column = analysis[
        "group_column"
    ]

    value_column = analysis[
        "value_column"
    ]

    aggregation = analysis[
        "aggregation"
    ]

    n = analysis[
        "n"
    ]


    # ========================================================
    # SINGLE VALUE
    # ========================================================

    if analysis_type == "single_value":

        if value_column == "order_id":

            result = (
                analysis_df[
                    "order_id"
                ]
                .nunique()
            )

            label = "Unique Orders"


        elif value_column is None:

            st.warning(
                "I couldn't identify the metric."
            )

            st.stop()


        else:

            series = (
                analysis_df[
                    value_column
                ]
                .dropna()
            )


            if aggregation == "sum":

                result = series.sum()

            elif aggregation == "mean":

                result = series.mean()

            elif aggregation == "count":

                result = series.count()

            elif aggregation == "min":

                result = series.min()

            elif aggregation == "max":

                result = series.max()

            else:

                result = series.sum()


            label = (
                f"{aggregation.title()} "
                f"{value_column.replace('_', ' ').title()}"
            )


        st.subheader(
            "🤖 Result"
        )


        st.success(
            f"The **{label.lower()}** is "
            f"**{result:,.2f}**."
        )


        st.metric(
            label,
            f"{result:,.2f}"
        )


    # ========================================================
    # GROUP BY
    # ========================================================

    elif analysis_type == "group_by":

        if (
            group_column is None
            or value_column is None
        ):

            st.warning(
                "I couldn't determine the grouping."
            )

            st.stop()


        grouped = (
            analysis_df
            .groupby(
                group_column,
                as_index=False
            )[value_column]
            .sum()
            .sort_values(
                value_column,
                ascending=False
            )
        )


        if grouped.empty:

            st.warning(
                "No results found."
            )

            st.stop()


        best = grouped.iloc[0]


        st.subheader(
            "🤖 AI Result"
        )


        st.success(
            f"**{best[group_column]}** has the highest "
            f"{value_column.replace('_', ' ')} "
            f"with **{best[value_column]:,.2f}**."
        )


        fig = px.bar(

            grouped,

            x=group_column,

            y=value_column,

            text_auto=".2s",

            title=(
                f"{value_column.title()} by "
                f"{group_column.replace('_', ' ').title()}"
            )
        )


        fig.update_layout(
            xaxis_title=(
                group_column
                .replace("_", " ")
                .title()
            ),
            yaxis_title=(
                value_column
                .replace("_", " ")
                .title()
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            grouped,
            use_container_width=True
        )


    # ========================================================
    # TOP N
    # ========================================================

    elif analysis_type == "top_n":

        if (
            group_column is None
            or value_column is None
        ):

            st.warning(
                "I couldn't determine the ranking."
            )

            st.stop()


        grouped = (
            analysis_df
            .groupby(
                group_column,
                as_index=False
            )[value_column]
            .sum()
            .sort_values(
                value_column,
                ascending=False
            )
            .head(n)
        )


        st.subheader(
            f"🏆 Top {n}"
        )


        fig = px.bar(

            grouped.sort_values(
                value_column
            ),

            x=value_column,

            y=group_column,

            orientation="h",

            text_auto=".2s",

            title=(
                f"Top {n} "
                f"{group_column.replace('_', ' ').title()} "
                f"by "
                f"{value_column.replace('_', ' ').title()}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            grouped,
            use_container_width=True
        )


    # ========================================================
    # BOTTOM N
    # ========================================================

    elif analysis_type == "bottom_n":

        if (
            group_column is None
            or value_column is None
        ):

            st.warning(
                "I couldn't determine the ranking."
            )

            st.stop()


        grouped = (
            analysis_df
            .groupby(
                group_column,
                as_index=False
            )[value_column]
            .sum()
            .sort_values(
                value_column,
                ascending=True
            )
            .head(n)
        )


        st.subheader(
            f"⚠️ Bottom {n}"
        )


        fig = px.bar(

            grouped,

            x=value_column,

            y=group_column,

            orientation="h",

            text_auto=".2s",

            title=(
                f"Bottom {n} "
                f"{group_column.replace('_', ' ').title()} "
                f"by "
                f"{value_column.replace('_', ' ').title()}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            grouped,
            use_container_width=True
        )


    # ========================================================
    # TREND
    # ========================================================

    elif analysis_type == "trend":

        if value_column is None:

            st.warning(
                "I couldn't identify the metric."
            )

            st.stop()


        if "Year" not in analysis_df.columns:

            st.warning(
                "Year information is unavailable."
            )

            st.stop()


        trend = (
            analysis_df
            .groupby(
                "Year",
                as_index=False
            )[value_column]
            .sum()
            .sort_values(
                "Year"
            )
        )


        st.subheader(
            "📈 Trend Analysis"
        )


        fig = px.line(

            trend,

            x="Year",

            y=value_column,

            markers=True,

            title=(
                f"{value_column.title()} Trend"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            trend,
            use_container_width=True
        )


    # ========================================================
    # COMPARISON
    # ========================================================

    elif analysis_type == "comparison":

        if (
            group_column is None
            or value_column is None
        ):

            st.warning(
                "I couldn't determine the comparison."
            )

            st.stop()


        comparison = (
            analysis_df
            .groupby(
                group_column,
                as_index=False
            )[value_column]
            .sum()
            .sort_values(
                value_column,
                ascending=False
            )
        )


        st.subheader(
            "📊 Comparison"
        )


        fig = px.bar(

            comparison,

            x=group_column,

            y=value_column,

            text_auto=".2s",

            title=(
                f"Comparison of "
                f"{value_column.title()}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            comparison,
            use_container_width=True
        )


    # ========================================================
    # NEGATIVE PROFIT
    # ========================================================

    elif analysis_type == "negative_profit":

        if "profit" not in analysis_df.columns:

            st.warning(
                "Profit column is unavailable."
            )

            st.stop()


        losses = (
            analysis_df[
                analysis_df["profit"] < 0
            ]
            .groupby(
                "product_name",
                as_index=False
            )["profit"]
            .sum()
            .sort_values(
                "profit",
                ascending=True
            )
        )


        st.subheader(
            "⚠️ Loss-Making Products"
        )


        st.metric(
            "Products with Negative Profit",
            f"{len(losses):,}"
        )


        fig = px.bar(

            losses.head(20),

            x="profit",

            y="product_name",

            orientation="h",

            title="Top 20 Loss-Making Products"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.dataframe(
            losses,
            use_container_width=True
        )


    # ========================================================
    # DISCOUNT VS PROFIT
    # ========================================================

    elif analysis_type == "discount_profit":

        if not all(
            col in analysis_df.columns
            for col in [
                "discount",
                "profit",
                "sales"
            ]
        ):

            st.warning(
                "Required columns are unavailable."
            )

            st.stop()


        discount_data = analysis_df[
            [
                "discount",
                "profit",
                "sales",
                "category",
                "region",
                "product_name"
            ]
        ].copy()


        discount_data = (
            discount_data
            .dropna(
                subset=[
                    "discount",
                    "profit",
                    "sales"
                ]
            )
        )


        discount_data = discount_data[
            discount_data["sales"] >= 0
        ]


        if len(discount_data) > 5000:

            discount_data = (
                discount_data
                .sample(
                    5000,
                    random_state=42
                )
            )


        st.subheader(
            "💸 Discount vs Profit"
        )


        fig = px.scatter(

            discount_data,

            x="discount",

            y="profit",

            size="sales",

            hover_data=[
                "product_name",
                "category",
                "region"
            ],

            title=(
                "Relationship Between "
                "Discount and Profit"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        correlation = (
            discount_data[
                ["discount", "profit"]
            ]
            .corr()
            .iloc[0, 1]
        )


        st.metric(
            "Discount-Profit Correlation",
            f"{correlation:.2f}"
        )


    # ========================================================
    # BUSINESS INSIGHTS
    # ========================================================

    elif analysis_type == "insights":

        st.subheader(
            "💡 Business Insights"
        )


        insights = []


        # --------------------------------------------
        # Sales
        # --------------------------------------------

        if all(
            c in analysis_df.columns
            for c in [
                "category",
                "sales"
            ]
        ):

            category_sales = (
                analysis_df
                .groupby("category")["sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )


            if not category_sales.empty:

                best_category = (
                    category_sales.index[0]
                )

                best_category_sales = (
                    category_sales.iloc[0]
                )


                insights.append(
                    f"**{best_category}** "
                    f"is the highest-sales category "
                    f"with sales of "
                    f"**{best_category_sales:,.2f}**."
                )


        # --------------------------------------------
        # Profit
        # --------------------------------------------

        if all(
            c in analysis_df.columns
            for c in [
                "category",
                "profit"
            ]
        ):

            category_profit = (
                analysis_df
                .groupby("category")["profit"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )


            if not category_profit.empty:

                best_profit_category = (
                    category_profit.index[0]
                )

                best_profit = (
                    category_profit.iloc[0]
                )


                insights.append(
                    f"**{best_profit_category}** "
                    f"generates the highest total profit "
                    f"of **{best_profit:,.2f}**."
                )


        # --------------------------------------------
        # Region
        # --------------------------------------------

        if all(
            c in analysis_df.columns
            for c in [
                "region",
                "profit"
            ]
        ):

            region_profit = (
                analysis_df
                .groupby("region")["profit"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )


            if not region_profit.empty:

                best_region = (
                    region_profit.index[0]
                )

                insights.append(
                    f"**{best_region}** "
                    f"is the most profitable region."
                )


        # --------------------------------------------
        # Discount
        # --------------------------------------------

        if all(
            c in analysis_df.columns
            for c in [
                "discount",
                "profit"
            ]
        ):

            corr = (
                analysis_df[
                    [
                        "discount",
                        "profit"
                    ]
                ]
                .dropna()
                .corr()
                .iloc[0, 1]
            )


            insights.append(
                f"The correlation between "
                f"discount and profit is "
                f"**{corr:.2f}**."
            )


        # --------------------------------------------
        # Negative profit
        # --------------------------------------------

        if "profit" in analysis_df.columns:

            negative_profit = (
                analysis_df["profit"] < 0
            ).sum()


            insights.append(
                f"There are **{negative_profit:,}** "
                f"rows with negative profit."
            )


        # --------------------------------------------
        # Display
        # --------------------------------------------

        for i, insight in enumerate(
            insights[:n],
            start=1
        ):

            st.markdown(
                f"### {i}. {insight}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Pandas • Streamlit • Hugging Face • "
    "Qwen3 • Plotly"
)
