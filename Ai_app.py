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

st.markdown(
    """
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

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🤖 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your e-commerce data in natural English.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "ecommerce_data.csv"
    )

    # ----------------------------
    # Dates
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
    # Numeric columns
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


try:

    df = load_data()

except Exception as e:

    st.error(
        "Could not load ecommerce_data.csv"
    )

    st.exception(e)

    st.stop()


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

    if "sales" in df.columns:

        c3.metric(
            "Total Sales",
            f"${df['sales'].sum():,.0f}"
        )

    else:

        c3.metric(
            "Total Sales",
            "N/A"
        )

    if "profit" in df.columns:

        c4.metric(
            "Total Profit",
            f"${df['profit'].sum():,.0f}"
        )

    else:

        c4.metric(
            "Total Profit",
            "N/A"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🤖 AI Data Analyst"
)

st.sidebar.caption(
    "Ask questions in normal English."
)

st.sidebar.divider()

st.sidebar.subheader(
    "💡 Example Questions"
)

example_questions = [

    "What is the total sales?",

    "Which category has the highest sales?",

    "Which category is most profitable?",

    "Which region has the highest sales?",

    "Which region has the highest profit?",

    "Show sales by category.",

    "Show profit by region.",

    "How did sales change over the years?",

    "What was the total profit in 2014?",

    "What was the sales in 2013?",

    "Show the top 5 products by sales.",

    "Show the top 10 products by profit.",

    "Show the worst 10 products by profit.",

    "Which products are losing money?",

    "What is the average discount?",

    "Does discount affect profit?",

    "Which segment has the highest sales?",

    "Give me 5 important business insights."

]

for question_text in example_questions:

    st.sidebar.write(
        "• " + question_text
    )


st.sidebar.divider()

st.sidebar.info(
    "Python/Pandas performs the actual calculations "
    "on the dataset."
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:

    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "HF_TOKEN is missing. "
        "Go to Manage app → Settings → Secrets "
        "and add HF_TOKEN."
    )

    st.stop()


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

try:

    client = InferenceClient(
        api_key=HF_TOKEN
    )

except Exception as e:

    st.error(
        "Could not initialize Hugging Face client."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json(text):

    if text is None:

        return None

    if not isinstance(text, str):

        return None

    text = text.strip()

    # Remove Qwen thinking section
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    # Remove markdown code fences
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

    # Find JSON object
    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1:

        return None

    json_text = text[
        start:end + 1
    ]

    try:

        return json.loads(
            json_text
        )

    except Exception:

        return None


# ============================================================
# SAFE INTEGER
# ============================================================

def safe_int(
    value,
    default=10
):

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

    if value > 100:

        return 100

    return value


# ============================================================
# NORMALIZE AI RESPONSE
# ============================================================

def normalize_analysis(
    result
):

    if not isinstance(
        result,
        dict
    ):

        result = {}


    analysis_type = result.get(
        "analysis_type",
        "single_value"
    )


    valid_analysis_types = [

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


    if (
        analysis_type
        not in valid_analysis_types
    ):

        analysis_type = "single_value"


    group_column = result.get(
        "group_column"
    )


    if (
        group_column is not None
        and group_column not in df.columns
    ):

        group_column = None


    value_column = result.get(
        "value_column"
    )


    if (
        value_column is not None
        and value_column not in df.columns
    ):

        value_column = None


    aggregation = result.get(
        "aggregation",
        "sum"
    )


    valid_aggregations = [

        "sum",

        "mean",

        "count",

        "min",

        "max"

    ]


    if (
        aggregation
        not in valid_aggregations
    ):

        aggregation = "sum"


    n = safe_int(
        result.get("n"),
        10
    )


    filter_year = result.get(
        "filter_year"
    )


    if filter_year in [
        None,
        "",
        "null",
        "None"
    ]:

        filter_year = None

    else:

        try:

            filter_year = int(
                filter_year
            )

        except (
            ValueError,
            TypeError
        ):

            filter_year = None


    return {

        "analysis_type":
            analysis_type,

        "group_column":
            group_column,

        "value_column":
            value_column,

        "aggregation":
            aggregation,

        "n":
            n,

        "filter_year":
            filter_year

    }


# ============================================================
# AI INTENT
# ============================================================

def get_ai_intent(
    question
):

    system_prompt = f"""

You are an expert Data Analyst.

Your ONLY job is to understand the user's question
and convert it into a JSON analysis instruction.

Do NOT calculate the answer.

Python/Pandas will calculate the real answer.

Dataset columns:

{json.dumps(
    list(df.columns),
    indent=2
)}

Numeric columns:

{json.dumps(
    [
        c
        for c in NUMERIC_COLUMNS
        if c in df.columns
    ],
    indent=2
)}

Grouping columns:

{json.dumps(
    [
        c
        for c in GROUP_COLUMNS
        if c in df.columns
    ],
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

Rules:

Sales, revenue, revenue generated
=> sales

Profit, profitability, earnings
=> profit

Quantity, units, items sold
=> quantity

Discount, discount percentage
=> discount

Shipping cost, delivery cost
=> shipping_cost

Orders, number of orders
=> order_id

Category
=> category

Sub-category
=> sub_category

Region
=> region

Segment
=> segment

Market
=> market

Country
=> country

State
=> state

Product
=> product_name

Customer
=> customer_name

If the user mentions a year,
put that year in filter_year.

Example:

User:
Which category has the highest sales?

JSON:

{{
    "analysis_type": "group_by",
    "group_column": "category",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

User:
Show top 5 products by sales.

JSON:

{{
    "analysis_type": "top_n",
    "group_column": "product_name",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 5,
    "filter_year": null
}}

User:
How did sales change over the years?

JSON:

{{
    "analysis_type": "trend",
    "group_column": "Year",
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

User:
Does discount affect profit?

JSON:

{{
    "analysis_type": "discount_profit",
    "group_column": null,
    "value_column": "profit",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null
}}

User:
Which products are losing money?

JSON:

{{
    "analysis_type": "negative_profit",
    "group_column": "product_name",
    "value_column": "profit",
    "aggregation": "sum",
    "n": 20,
    "filter_year": null
}}

User:
Give me 5 important business insights.

JSON:

{{
    "analysis_type": "insights",
    "group_column": null,
    "value_column": null,
    "aggregation": "sum",
    "n": 5,
    "filter_year": null
}}

Never calculate numbers.

Never invent columns.

Return JSON only.

"""


    user_prompt = f"""

User question:

{question}

Return ONLY JSON.

"""


    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",
                        "content":
                            system_prompt
                    },

                    {
                        "role": "user",
                        "content":
                            user_prompt
                    }

                ],

                max_tokens=500,

                temperature=0.0
            )
        )


        raw = (
            response
            .choices[0]
            .message
            .content
        )


        # TEMPORARY DEBUG
        st.write(
            "### 🔍 Raw Qwen Response"
        )

        st.code(
            raw
        )


        parsed = clean_json(
            raw
        )


        return parsed


    except Exception as e:

        st.error(
            "Hugging Face / Qwen error:"
        )

        st.exception(
            e
        )

        return None


# ============================================================
# APPLY FILTERS
# ============================================================

def apply_filters(
    data,
    analysis
):

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

st.subheader(
    "💬 Ask Your Data"
)


question = st.text_input(

    "Write your question in English:",

    placeholder=(
        "Example: Which category has "
        "the highest sales?"
    )
)


analyze_button = st.button(

    "🔍 Analyze",

    type="primary"
)


# ============================================================
# RUN
# ============================================================

if analyze_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    with st.spinner(
        "🤖 AI is understanding your question..."
    ):

        analysis = get_ai_intent(
            question
        )


    if analysis is None:

        st.error(
            "Qwen did not return valid JSON."
        )

        st.info(
            "Check the Raw Qwen Response above."
        )

        st.stop()


    analysis = normalize_analysis(
        analysis
    )


    analysis_df = apply_filters(
        df,
        analysis
    )


    if analysis_df.empty:

        st.warning(
            "No data found for the selected filter."
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
            f"**{best[group_column]}** has the "
            f"highest {value_column.replace('_', ' ')} "
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

        if (
            value_column is None
            or "Year" not in analysis_df.columns
        ):

            st.warning(
                "Year or metric information is unavailable."
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
                f"{value_column.title()} Comparison"
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

        if (
            "profit" not in analysis_df.columns
            or "product_name"
            not in analysis_df.columns
        ):

            st.warning(
                "Required columns are unavailable."
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
            "Loss-Making Products",
            f"{len(losses):,}"
        )


        fig = px.bar(

            losses.head(20),

            x="profit",

            y="product_name",

            orientation="h",

            title=(
                "Top 20 Loss-Making Products"
            )
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

        required = [
            "discount",
            "profit",
            "sales"
        ]


        if not all(
            col in analysis_df.columns
            for col in required
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
                "product_name",
                "category",
                "region"
            ]
        ].copy()


        for column in [
            "discount",
            "profit",
            "sales"
        ]:

            discount_data[column] = pd.to_numeric(
                discount_data[column],
                errors="coerce"
            )


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
                [
                    "discount",
                    "profit"
                ]
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
        # Category Sales
        # --------------------------------------------

        if all(
            column in analysis_df.columns
            for column in [
                "category",
                "sales"
            ]
        ):

            category_sales = (
                analysis_df
                .groupby(
                    "category"
                )["sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )


            if not category_sales.empty:

                best_category = (
                    category_sales.index[0]
                )

                best_sales = (
                    category_sales.iloc[0]
                )


                insights.append(
                    f"**{best_category}** is the "
                    f"highest-sales category with "
                    f"sales of **{best_sales:,.2f}**."
                )


        # --------------------------------------------
        # Category Profit
        # --------------------------------------------

        if all(
            column in analysis_df.columns
            for column in [
                "category",
                "profit"
            ]
        ):

            category_profit = (
                analysis_df
                .groupby(
                    "category"
                )["profit"]
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
                    f"generates the highest total "
                    f"profit of "
                    f"**{best_profit:,.2f}**."
                )


        # --------------------------------------------
        # Region
        # --------------------------------------------

        if all(
            column in analysis_df.columns
            for column in [
                "region",
                "profit"
            ]
        ):

            region_profit = (
                analysis_df
                .groupby(
                    "region"
                )["profit"]
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
                    f"**{best_region}** is the "
                    f"most profitable region."
                )


        # --------------------------------------------
        # Discount
        # --------------------------------------------

        if all(
            column in analysis_df.columns
            for column in [
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
        # Negative Profit
        # --------------------------------------------

        if "profit" in analysis_df.columns:

            negative_rows = (
                analysis_df["profit"] < 0
            ).sum()


            insights.append(
                f"There are **{negative_rows:,}** "
                f"rows with negative profit."
            )


        # --------------------------------------------
        # Display
        # --------------------------------------------

        if not insights:

            st.info(
                "No insights could be generated."
            )

        else:

            for index, insight in enumerate(
                insights[:n],
                start=1
            ):

                st.markdown(
                    f"### {index}. {insight}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Pandas • Streamlit • "
    "Hugging Face • Qwen3 • Plotly"
)
