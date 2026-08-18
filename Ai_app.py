import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import json
import re

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
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
    'Ask questions about your e-commerce data in plain English.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ecommerce_data.csv")

    # Dates
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

    if "ship_date" in df.columns:
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
    if "order_date" in df.columns:

        df["Year"] = df["order_date"].dt.year

    return df


df = load_data()

# ============================================================
# DATA INFORMATION
# ============================================================

with st.expander("📊 Dataset Information"):

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🤖 AI Data Analyst")

st.sidebar.markdown(
    "Ask questions about your dataset."
)

st.sidebar.divider()

st.sidebar.subheader("💡 Example Questions")

examples = [
    "What is the total sales?",
    "Which category has the highest sales?",
    "Which region generated the highest profit?",
    "Show sales by category.",
    "What was the total profit in 2014?",
    "Show the top 10 products by sales.",
    "Which products have negative profit?"
]

for question in examples:

    st.sidebar.write(
        "• " + question
    )

# ============================================================
# API KEY
# ============================================================

try:

    api_key = st.secrets["OPENAI_API_KEY"]

except Exception:

    st.error(
        "OpenAI API key is not configured. "
        "Add OPENAI_API_KEY in Streamlit Secrets."
    )

    st.stop()

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key
)

# ============================================================
# DATA SUMMARY FOR AI
# ============================================================

def create_data_summary(data):

    summary = {}

    summary["columns"] = list(data.columns)

    summary["rows"] = len(data)

    summary["data_types"] = {
        col: str(dtype)
        for col, dtype in data.dtypes.items()
    }

    summary["numeric_summary"] = {}

    numeric_cols = data.select_dtypes(
        include="number"
    ).columns

    for col in numeric_cols:

        summary["numeric_summary"][col] = {
            "sum": float(
                data[col].sum()
            ),
            "mean": float(
                data[col].mean()
            ) if not data[col].dropna().empty else 0,
            "min": float(
                data[col].min()
            ) if not data[col].dropna().empty else 0,
            "max": float(
                data[col].max()
            ) if not data[col].dropna().empty else 0
        }

    # Unique categorical values
    summary["categorical_values"] = {}

    categorical_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:

        values = data[col].dropna().unique()

        summary["categorical_values"][col] = [
            str(x)
            for x in values[:50]
        ]

    return summary


# ============================================================
# AI QUESTION
# ============================================================

st.subheader("💬 Ask Your Data")

question = st.text_input(
    "Type your question in English:",
    placeholder="Example: Which category has the highest sales?"
)

analyze = st.button(
    "🔍 Analyze Data",
    type="primary"
)

# ============================================================
# AI ANALYSIS
# ============================================================

if analyze:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()

    with st.spinner(
        "🤖 AI is analyzing your data..."
    ):

        try:

            data_summary = create_data_summary(df)

            system_prompt = """
You are an expert Data Analyst.

You are analyzing an e-commerce dataset.

The dataset contains columns such as:
order_id, order_date, ship_date, ship_mode,
customer_name, segment, state, country, market,
region, product_id, category, sub_category,
product_name, sales, quantity, discount, profit,
shipping_cost and Year.

Your job is to understand the user's English question
and determine what analysis is required.

Return ONLY valid JSON.

Use this exact structure:

{
    "answer": "short natural language answer",
    "analysis_type": "single_value OR group_by OR top_n OR bottom_n OR table",
    "group_column": "column name or null",
    "value_column": "column name or null",
    "aggregation": "sum OR mean OR count OR min OR max OR null",
    "n": 10
}

Rules:

1. Use only columns that exist in the dataset.
2. Sales questions normally use sales.
3. Profit questions normally use profit.
4. Quantity questions normally use quantity.
5. Revenue means sales.
6. Orders means unique order_id.
7. If user asks "by category", group_column should be category.
8. If user asks "by region", group_column should be region.
9. If user asks for top products, use product_name.
10. If user asks for highest/lowest single result, use group_by.
11. If user asks for top 10, use top_n.
12. If user asks for bottom 10, use bottom_n.
13. Keep the answer concise.
14. Do not invent columns.
"""

            user_prompt = f"""
Dataset summary:

{json.dumps(
    data_summary,
    indent=2,
    default=str
)}

User question:

{question}
"""

            response = client.responses.create(
                model="gpt-5.6-luna",
                instructions=system_prompt,
                input=user_prompt
            )

            ai_text = response.output_text.strip()

            # Remove markdown code fences if AI returns them
            ai_text = re.sub(
                r"```json|```",
                "",
                ai_text
            ).strip()

            analysis = json.loads(
                ai_text
            )

            # ==================================================
            # EXTRACT AI INSTRUCTIONS
            # ==================================================

            answer = analysis.get(
                "answer",
                ""
            )

            analysis_type = analysis.get(
                "analysis_type",
                "single_value"
            )

            group_column = analysis.get(
                "group_column"
            )

            value_column = analysis.get(
                "value_column"
            )

            aggregation = analysis.get(
                "aggregation"
            )

            n = analysis.get(
                "n",
                10
            )

            # ==================================================
            # VALIDATE COLUMNS
            # ==================================================

            if (
                group_column
                and group_column not in df.columns
            ):

                group_column = None

            if (
                value_column
                and value_column not in df.columns
            ):

                value_column = None

            # ==================================================
            # DISPLAY ANSWER
            # ==================================================

            st.subheader("🤖 AI Answer")

            st.markdown(
                f"""
                <div class="answer-box">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            # ==================================================
            # SINGLE VALUE
            # ==================================================

            if analysis_type == "single_value":

                if (
                    value_column
                    and value_column in df.columns
                ):

                    if aggregation == "sum":

                        value = df[
                            value_column
                        ].sum()

                    elif aggregation == "mean":

                        value = df[
                            value_column
                        ].mean()

                    elif aggregation == "min":

                        value = df[
                            value_column
                        ].min()

                    elif aggregation == "max":

                        value = df[
                            value_column
                        ].max()

                    elif aggregation == "count":

                        value = df[
                            value_column
                        ].count()

                    else:

                        value = df[
                            value_column
                        ].sum()

                    st.metric(
                        value_column.title(),
                        f"{value:,.2f}"
                    )

            # ==================================================
            # GROUP BY
            # ==================================================

            elif (
                analysis_type == "group_by"
                and group_column
                and value_column
            ):

                grouped = (
                    df
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
                    f"📊 {value_column.title()} by "
                    f"{group_column.replace('_', ' ').title()}"
                )

                fig = px.bar(
                    grouped,
                    x=group_column,
                    y=value_column,
                    text_auto=".2s"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

            # ==================================================
            # TOP N
            # ==================================================

            elif (
                analysis_type == "top_n"
                and group_column
                and value_column
            ):

                grouped = (
                    df
                    .groupby(
                        group_column,
                        as_index=False
                    )[value_column]
                    .sum()
                    .sort_values(
                        value_column,
                        ascending=False
                    )
                    .head(int(n))
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
                    text_auto=".2s"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

            # ==================================================
            # BOTTOM N
            # ==================================================

            elif (
                analysis_type == "bottom_n"
                and group_column
                and value_column
            ):

                grouped = (
                    df
                    .groupby(
                        group_column,
                        as_index=False
                    )[value_column]
                    .sum()
                    .sort_values(
                        value_column,
                        ascending=True
                    )
                    .head(int(n))
                )

                st.subheader(
                    f"⚠️ Bottom {n}"
                )

                fig = px.bar(
                    grouped,
                    x=value_column,
                    y=group_column,
                    orientation="h",
                    text_auto=".2s"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

            # ==================================================
            # TABLE
            # ==================================================

            elif (
                analysis_type == "table"
                and group_column
                and value_column
            ):

                grouped = (
                    df
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

                st.dataframe(
                    grouped,
                    use_container_width=True
                )

        except json.JSONDecodeError:

            st.error(
                "AI returned an unexpected response. "
                "Please try asking the question again."
            )

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Pandas • Streamlit • OpenAI • Plotly"
)
