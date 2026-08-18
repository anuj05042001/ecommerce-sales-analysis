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
# SETTINGS
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
# HEADER
# ============================================================

st.markdown(
    """
    <h1>🤖 AI Data Analyst</h1>

    <p style="font-size:18px;">
    Ask questions about your e-commerce data
    using normal English.
    </p>
    """,
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
# DATASET INFORMATION
# ============================================================

with st.expander("📊 Dataset Information"):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    if "sales" in df.columns:

        col3.metric(
            "Total Sales",
            f"${df['sales'].sum():,.0f}"
        )

    if "profit" in df.columns:

        col4.metric(
            "Total Profit",
            f"${df['profit'].sum():,.0f}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "💡 Example Questions"
)

example_questions = [

    "What is the total sales?",

    "What is the total profit?",

    "Which category has the highest sales?",

    "Which category has the highest profit?",

    "Which region has the highest sales?",

    "Which region has the highest profit?",

    "Show sales by category.",

    "Show profit by region.",

    "Show sales by segment.",

    "How did sales change over the years?",

    "How did profit change over the years?",

    "What was the total sales in 2014?",

    "What was the total profit in 2013?",

    "Show the top 5 products by sales.",

    "Show the top 10 products by profit.",

    "Show the worst 10 products by profit.",

    "Which products are losing money?",

    "What is the average discount?",

    "Does discount affect profit?",

    "Which segment has the highest sales?",

    "Give me 5 important business insights."

]

for q in example_questions:

    st.sidebar.write(
        "• " + q
    )


st.sidebar.divider()

st.sidebar.info(
    "The AI understands the question. "
    "Python/Pandas performs the actual calculations."
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:

    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        """
        ❌ HF_TOKEN is not configured.

        Go to:

        Manage app → Settings → Secrets

        Add:

        HF_TOKEN = "your_token_here"
        """
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

    if not isinstance(
        text,
        str
    ):

        return None

    text = text.strip()


    # Remove Qwen thinking blocks

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )


    # Remove markdown code blocks

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```JSON",
        ""
    )

    text = text.replace(
        "```",
        ""
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

        # Try fixing common JSON problems

        json_text = re.sub(
            r",\s*}",
            "}",
            json_text
        )

        json_text = re.sub(
            r",\s*]",
            "]",
            json_text
        )

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

    try:

        if value is None:

            return default

        value = int(value)


        if value < 1:

            return default


        if value > 100:

            return 100


        return value


    except (
        ValueError,
        TypeError
    ):

        return default


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


    if aggregation not in valid_aggregations:

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
        "None",
        "null"
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
# GET AI INTENT
# ============================================================

def get_ai_intent(
    question
):

    system_prompt = f"""

You are an expert Data Analyst.

Your ONLY job is to understand the user's question
and convert it into a JSON analysis instruction.

DO NOT calculate the answer.

Python/Pandas will calculate the answer.

Return ONLY valid JSON.

Do NOT write explanations.

Do NOT write markdown.

Do NOT use code fences.

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


Use this exact structure:

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


Column mapping:

sales / revenue
= sales

profit / earnings
= profit

quantity / units
= quantity

discount
= discount

shipping cost
= shipping_cost

category
= category

sub category
= sub_category

region
= region

segment
= segment

market
= market

country
= country

state
= state

product
= product_name

customer
= customer_name

orders
= order_id


EXAMPLE 1

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


EXAMPLE 2

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


EXAMPLE 3

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


EXAMPLE 4

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


EXAMPLE 5

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


EXAMPLE 6

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


If the user mentions a year,
put the year in filter_year.

Never invent columns.

Return JSON only.

"""


    user_prompt = f"""

Convert this user question
into the required JSON:

{question}

Return ONLY JSON.

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


        # --------------------------------------------
        # RESPONSE CHECK
        # --------------------------------------------

        if response is None:

            st.error(
                "Hugging Face returned an empty response."
            )

            return None


        # --------------------------------------------
        # CHOICES
        # --------------------------------------------

        choices = getattr(
            response,
            "choices",
            None
        )


        if not choices:

            st.error(
                "Qwen returned no choices."
            )

            st.write(
                response
            )

            return None


        # --------------------------------------------
        # MESSAGE
        # --------------------------------------------

        message = getattr(
            choices[0],
            "message",
            None
        )


        if message is None:

            st.error(
                "Qwen returned no message."
            )

            st.write(
                response
            )

            return None


        # --------------------------------------------
        # NORMAL CONTENT
        # --------------------------------------------

        raw = getattr(
            message,
            "content",
            None
        )


        # --------------------------------------------
        # REASONING FALLBACK
        # --------------------------------------------

        if not raw:

            raw = getattr(
                message,
                "reasoning_content",
                None
            )


        # --------------------------------------------
        # FINAL CHECK
        # --------------------------------------------

        if not raw:

            st.error(
                "Qwen returned no text content."
            )

            st.write(
                "Full response:"
            )

            st.write(
                response
            )

            return None


        # --------------------------------------------
        # DEBUG RESPONSE
        # --------------------------------------------

        with st.expander(
            "🔍 View AI Response"
        ):

            st.code(
                raw
            )


        # --------------------------------------------
        # PARSE JSON
        # --------------------------------------------

        parsed = clean_json(
            raw
        )


        return parsed


    except Exception as e:

        st.error(
            "Hugging Face / Qwen error"
        )

        st.exception(
            e
        )

        return None


# ============================================================
# APPLY YEAR FILTER
# ============================================================

def apply_filters(
    data,
    analysis
):

    result = data.copy()


    year = analysis.get(
        "filter_year"
    )


    if (
        year is not None
        and "Year" in result.columns
    ):

        result = result[
            result["Year"] == year
        ]


    return result


# ============================================================
# MAIN QUESTION BOX
# ============================================================

st.subheader(
    "💬 Ask Your Data"
)


question = st.text_input(

    "Write your question in English",

    placeholder=(
        "Example: Which category has "
        "the highest sales?"
    )
)


analyze = st.button(
    "🔍 Analyze",
    type="primary"
)


# ============================================================
# ANALYZE
# ============================================================

if analyze:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ========================================================
    # AI
    # ========================================================

    with st.spinner(
        "🤖 AI is understanding your question..."
    ):

        analysis = get_ai_intent(
            question
        )


    if analysis is None:

        st.error(
            "I couldn't understand the AI response."
        )

        st.info(
            "Check the AI Response section above."
        )

        st.stop()


    # ========================================================
    # NORMALIZE
    # ========================================================

    analysis = normalize_analysis(
        analysis
    )


    # ========================================================
    # FILTER
    # ========================================================

    filtered_df = apply_filters(
        df,
        analysis
    )


    if filtered_df.empty:

        st.warning(
            "No data found for this question."
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

    n = analysis[
        "n"
    ]


    # ========================================================
    # SINGLE VALUE
    # ========================================================

    if analysis_type == "single_value":

        if value_column is None:

            st.warning(
                "I couldn't identify the metric."
            )

            st.stop()


        if value_column == "order_id":

            result = (
                filtered_df[
                    "order_id"
                ]
                .nunique()
            )

        else:

            series = (
                filtered_df[
                    value_column
                ]
                .dropna()
            )


            aggregation = analysis[
                "aggregation"
            ]


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


        st.subheader(
            "🤖 Answer"
        )


        st.success(
            f"The {value_column.replace('_', ' ')} "
            f"is **{result:,.2f}**."
        )


        st.metric(
            value_column.replace(
                "_",
                " "
            ).title(),
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
            filtered_df
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
            filtered_df
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
                f"{group_column.replace('_', ' ')} "
                f"by "
                f"{value_column.replace('_', ' ')}"
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
            filtered_df
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
                f"{group_column.replace('_', ' ')}"
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
            "Year" not in filtered_df.columns
            or value_column is None
        ):

            st.warning(
                "Year information is unavailable."
            )

            st.stop()


        trend = (
            filtered_df
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
            "📈 Yearly Trend"
        )


        fig = px.line(

            trend,

            x="Year",

            y=value_column,

            markers=True,

            title=(
                f"{value_column.title()} "
                f"Trend Over Years"
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
                "Comparison information is missing."
            )

            st.stop()


        comparison = (
            filtered_df
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
            "profit" not in filtered_df.columns
            or "product_name"
            not in filtered_df.columns
        ):

            st.warning(
                "Required columns are missing."
            )

            st.stop()


        losses = (
            filtered_df[
                filtered_df["profit"] < 0
            ]
            .groupby(
                "product_name",
                as_index=False
            )["profit"]
            .sum()
            .sort_values(
                "profit"
            )
            .head(n)
        )


        st.subheader(
            "⚠️ Loss-Making Products"
        )


        fig = px.bar(

            losses,

            x="profit",

            y="product_name",

            orientation="h",

            title="Loss-Making Products"
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

        required_columns = [

            "discount",
            "profit",
            "sales"

        ]


        if not all(
            column in filtered_df.columns
            for column in required_columns
        ):

            st.warning(
                "Required columns are missing."
            )

            st.stop()


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


        # Category Sales
        if all(
            c in filtered_df.columns
            for c in [
                "category",
                "sales"
            ]
        ):

            category_sales = (

                filtered_df

                .groupby(
                    "category"
                )["sales"]

                .sum()

                .sort_values(
                    ascending=False
                )

            )


            if not category_sales.empty:

                insights.append(

                    f"**{category_sales.index[0]}** "
                    f"has the highest sales with "
                    f"**{category_sales.iloc[0]:,.2f}**."

                )


        # Category Profit
        if all(
            c in filtered_df.columns
            for c in [
                "category",
                "profit"
            ]
        ):

            category_profit = (

                filtered_df

                .groupby(
                    "category"
                )["profit"]

                .sum()

                .sort_values(
                    ascending=False
                )

            )


            if not category_profit.empty:

                insights.append(

                    f"**{category_profit.index[0]}** "
                    f"is the most profitable category "
                    f"with profit of "
                    f"**{category_profit.iloc[0]:,.2f}**."

                )


        # Region
        if all(
            c in filtered_df.columns
            for c in [
                "region",
                "profit"
            ]
        ):

            region_profit = (

                filtered_df

                .groupby(
                    "region"
                )["profit"]

                .sum()

                .sort_values(
                    ascending=False
                )

            )


            if not region_profit.empty:

                insights.append(

                    f"**{region_profit.index[0]}** "
                    f"is the most profitable region."

                )


        # Discount
        if all(
            c in filtered_df.columns
            for c in [
                "discount",
                "profit"
            ]
        ):

            correlation = (

                filtered_df[
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

                f"Discount and profit have a "
                f"correlation of "
                f"**{correlation:.2f}**."

            )


        # Loss records
        if "profit" in filtered_df.columns:

            loss_records = (

                filtered_df["profit"] < 0

            ).sum()


            insights.append(

                f"There are **{loss_records:,}** "
                f"records with negative profit."

            )


        # Display
        if not insights:

            st.info(
                "No insights available."
            )

        else:

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
    "Built with Python • Pandas • Streamlit • "
    "Hugging Face • Qwen3 • Plotly"
)
