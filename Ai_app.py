import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient
import json
import re


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

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


# =========================================================
# TITLE
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ecommerce_data.csv")

    # Convert dates
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

    # Convert numeric columns
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

    # Create Year
    if "order_date" in df.columns:

        df["Year"] = df["order_date"].dt.year

    return df


df = load_data()


# =========================================================
# DATA INFORMATION
# =========================================================

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


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 AI Data Analyst")

st.sidebar.write(
    "Ask questions about your dataset."
)

st.sidebar.divider()

st.sidebar.subheader("💡 Try these questions")

example_questions = [
    "What is the total sales?",
    "Which category has the highest sales?",
    "Which region generated the highest profit?",
    "Show sales by category.",
    "What was the total profit in 2014?",
    "Show the top 10 products by sales.",
    "Which products have negative profit?"
]

for q in example_questions:

    st.sidebar.write("• " + q)


# =========================================================
# HUGGING FACE TOKEN
# =========================================================

try:

    hf_token = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "Hugging Face token is not configured. "
        "Add HF_TOKEN in Streamlit Secrets."
    )

    st.stop()


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    api_key=hf_token
)


# =========================================================
# DATA SUMMARY
# =========================================================

def create_data_summary(data):

    summary = {}

    summary["rows"] = len(data)

    summary["columns"] = list(data.columns)

    summary["data_types"] = {
        col: str(dtype)
        for col, dtype in data.dtypes.items()
    }

    summary["numeric_columns"] = []

    for col in data.select_dtypes(
        include="number"
    ).columns:

        summary["numeric_columns"].append(col)

    summary["categorical_columns"] = []

    for col in data.select_dtypes(
        include=["object", "category"]
    ).columns:

        summary["categorical_columns"].append(col)

    return summary


data_summary = create_data_summary(df)


# =========================================================
# USER QUESTION
# =========================================================

st.subheader("💬 Ask Your Data")

question = st.text_input(
    "Type your question in English:",
    placeholder="Example: Which category has the highest sales?"
)


analyze_button = st.button(
    "🔍 Analyze Data",
    type="primary"
)


# =========================================================
# AI ANALYSIS
# =========================================================

if analyze_button:

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()


    with st.spinner(
        "🤖 AI is analyzing your question..."
    ):

        try:

            # -------------------------------------------------
            # AI SYSTEM PROMPT
            # -------------------------------------------------

            system_prompt = """
You are an expert Data Analyst.

You analyze an e-commerce dataset.

Available columns are:

order_id
order_date
ship_date
ship_mode
customer_name
segment
state
country
market
region
product_id
category
sub_category
product_name
sales
quantity
discount
profit
shipping_cost
Year

The user will ask a question in normal English.

Your job is to determine the required analysis.

Return ONLY valid JSON.

Use exactly this format:

{
    "analysis_type": "single_value",
    "group_column": null,
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null,
    "answer": "Short explanation"
}

Allowed analysis_type values:

single_value
group_by
top_n
bottom_n

Allowed aggregation values:

sum
mean
count
min
max

Rules:

1. "Sales" or "revenue" means sales.
2. "Profit" means profit.
3. "Quantity" means quantity.
4. "Orders" means unique order_id.
5. "By category" means group_column = category.
6. "By region" means group_column = region.
7. "By segment" means group_column = segment.
8. "By sub-category" means group_column = sub_category.
9. "By product" means group_column = product_name.
10. "Top 10 products" means analysis_type = top_n.
11. "Bottom 10 products" means analysis_type = bottom_n.
12. If the question contains a year such as 2014, set filter_year to that year.
13. Do not invent column names.
14. Keep answer short.
"""


            # -------------------------------------------------
            # USER PROMPT
            # -------------------------------------------------

            user_prompt = f"""
Dataset information:

{json.dumps(data_summary, indent=2)}

User question:

{question}
"""


            # -------------------------------------------------
            # HUGGING FACE REQUEST
            # -------------------------------------------------

            response = client.chat.completions.create(

                model="Qwen/Qwen2.5-7B-Instruct",

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

                temperature=0.1
            )


            # -------------------------------------------------
            # GET AI RESPONSE
            # -------------------------------------------------

            ai_text = response.choices[
                0
            ].message.content.strip()


            # Remove markdown fences
            ai_text = re.sub(
                r"```json|```",
                "",
                ai_text
            ).strip()


            analysis = json.loads(
                ai_text
            )


            # -------------------------------------------------
            # READ AI INSTRUCTIONS
            # -------------------------------------------------

            analysis_type = analysis.get(
                "analysis_type"
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

            n = int(
                analysis.get(
                    "n",
                    10
                )
            )

            filter_year = analysis.get(
                "filter_year"
            )


            # -------------------------------------------------
            # APPLY YEAR FILTER
            # -------------------------------------------------

            analysis_df = df.copy()

            if filter_year:

                analysis_df = analysis_df[
                    analysis_df["Year"]
                    == int(filter_year)
                ]


            # -------------------------------------------------
            # VALIDATE COLUMNS
            # -------------------------------------------------

            if (
                group_column
                and group_column not in df.columns
            ):

                st.error(
                    "AI selected an invalid group column."
                )

                st.stop()


            if (
                value_column
                and value_column not in df.columns
            ):

                st.error(
                    "AI selected an invalid value column."
                )

                st.stop()


            # =================================================
            # SINGLE VALUE
            # =================================================

            if analysis_type == "single_value":

                if value_column == "order_id":

                    result = analysis_df[
                        "order_id"
                    ].nunique()

                elif aggregation == "sum":

                    result = analysis_df[
                        value_column
                    ].sum()

                elif aggregation == "mean":

                    result = analysis_df[
                        value_column
                    ].mean()

                elif aggregation == "min":

                    result = analysis_df[
                        value_column
                    ].min()

                elif aggregation == "max":

                    result = analysis_df[
                        value_column
                    ].max()

                elif aggregation == "count":

                    result = analysis_df[
                        value_column
                    ].count()

                else:

                    result = analysis_df[
                        value_column
                    ].sum()


                st.subheader("🤖 AI Answer")

                st.success(
                    f"{analysis.get('answer', '')}"
                )


                st.metric(
                    value_column.replace(
                        "_",
                        " "
                    ).title(),

                    f"{result:,.2f}"
                )


            # =================================================
            # GROUP BY
            # =================================================

            elif (
                analysis_type == "group_by"
                and group_column
                and value_column
            ):

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


                st.subheader(
                    "🤖 AI Answer"
                )

                st.success(
                    analysis.get(
                        "answer",
                        "Here is the requested analysis."
                    )
                )


                fig = px.bar(
                    grouped,
                    x=group_column,
                    y=value_column,
                    text_auto=".2s",
                    title=(
                        f"{value_column.title()} "
                        f"by "
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


            # =================================================
            # TOP N
            # =================================================

            elif (
                analysis_type == "top_n"
                and group_column
                and value_column
            ):

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
                    "🏆 Top Results"
                )


                fig = px.bar(
                    grouped.sort_values(
                        value_column
                    ),
                    x=value_column,
                    y=group_column,
                    orientation="h",
                    text_auto=".2s",
                    title=f"Top {n}"
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.dataframe(
                    grouped,
                    use_container_width=True
                )


            # =================================================
            # BOTTOM N
            # =================================================

            elif (
                analysis_type == "bottom_n"
                and group_column
                and value_column
            ):

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
                    "⚠️ Bottom Results"
                )


                fig = px.bar(
                    grouped,
                    x=value_column,
                    y=group_column,
                    orientation="h",
                    text_auto=".2s",
                    title=f"Bottom {n}"
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


                st.dataframe(
                    grouped,
                    use_container_width=True
                )


            else:

                st.warning(
                    "I could not determine the required analysis."
                )


        except json.JSONDecodeError:

            st.error(
                "The AI returned an invalid response. "
                "Please try the question again."
            )

            st.code(
                ai_text
            )


        except Exception as e:

            st.error(
                "Something went wrong while analyzing the data."
            )

            st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Pandas • Streamlit • Hugging Face • Plotly"
)
