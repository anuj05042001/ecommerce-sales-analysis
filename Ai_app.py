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

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    if "order_date" in df.columns:

        df["Year"] = (
            df["order_date"]
            .dt.year
        )

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

st.sidebar.title("🤖 AI Data Analyst")

st.sidebar.write(
    "Ask questions about your e-commerce dataset."
)

st.sidebar.divider()

st.sidebar.subheader("💡 Example Questions")

example_questions = [

    "What is the total sales?",

    "Which category has the highest sales?",

    "Which region generated the highest profit?",

    "Show sales by category.",

    "Show profit by region.",

    "What was the total profit in 2014?",

    "What was the sales in 2013?",

    "Show the top 10 products by sales.",

    "Show the bottom 10 products by profit.",

    "Which category has the highest profit?",

    "Which segment has the highest sales?",

    "Compare sales between categories.",

    "Which products have negative profit?"

]

for q in example_questions:

    st.sidebar.write(
        "• " + q
    )


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:

    HF_TOKEN = st.secrets["HF_TOKEN"]

except Exception:

    st.error(
        "Hugging Face token is not configured. "
        "Add HF_TOKEN in Streamlit Secrets."
    )

    st.stop()


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_dataset_summary(data):

    return {
        "rows": len(data),
        "columns": list(data.columns),
        "numeric_columns": list(
            data.select_dtypes(
                include="number"
            ).columns
        ),
        "categorical_columns": list(
            data.select_dtypes(
                include=["object", "category"]
            ).columns
        )
    }


dataset_summary = get_dataset_summary(df)


# ============================================================
# CLEAN AI JSON
# ============================================================

def clean_json_response(text):

    if not text:
        return None

    text = text.strip()

    # Remove <think>...</think> if model produces reasoning
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()

    # Remove markdown fences
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

    # Find JSON object if extra text exists
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        text = text[start:end + 1]

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        return None


# ============================================================
# NORMALIZE AI RESPONSE
# ============================================================

def normalize_analysis(analysis):

    if not isinstance(analysis, dict):

        analysis = {}

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

    n_value = analysis.get(
        "n"
    )

    filter_year = analysis.get(
        "filter_year"
    )

    filter_category = analysis.get(
        "filter_category"
    )

    filter_region = analysis.get(
        "filter_region"
    )

    # --------------------------------------------------------
    # Safe n
    # --------------------------------------------------------

    if n_value is None:

        n = 10

    else:

        try:

            n = int(n_value)

        except (
            ValueError,
            TypeError
        ):

            n = 10

    if n <= 0:

        n = 10

    if n > 100:

        n = 100

    # --------------------------------------------------------
    # Validate analysis type
    # --------------------------------------------------------

    valid_types = [
        "single_value",
        "group_by",
        "top_n",
        "bottom_n",
        "negative_profit"
    ]

    if analysis_type not in valid_types:

        analysis_type = "single_value"

    # --------------------------------------------------------
    # Validate aggregation
    # --------------------------------------------------------

    valid_aggregations = [
        "sum",
        "mean",
        "count",
        "min",
        "max"
    ]

    if aggregation not in valid_aggregations:

        aggregation = "sum"

    # --------------------------------------------------------
    # Validate group column
    # --------------------------------------------------------

    if (
        group_column is not None
        and group_column not in df.columns
    ):

        group_column = None

    # --------------------------------------------------------
    # Validate value column
    # --------------------------------------------------------

    if (
        value_column is not None
        and value_column not in df.columns
    ):

        value_column = None

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    if filter_year in [
        "",
        "null",
        "None"
    ]:

        filter_year = None

    if filter_year is not None:

        try:

            filter_year = int(filter_year)

        except (
            ValueError,
            TypeError
        ):

            filter_year = None

    return {
        "analysis_type": analysis_type,
        "group_column": group_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "n": n,
        "filter_year": filter_year,
        "filter_category": filter_category,
        "filter_region": filter_region
    }


# ============================================================
# AI QUESTION
# ============================================================

st.subheader("💬 Ask Your Data")

question = st.text_input(
    "Ask a question in English:",
    placeholder=(
        "Example: Which category has the highest sales?"
    )
)

analyze_button = st.button(
    "🔍 Analyze Data",
    type="primary"
)


# ============================================================
# MAIN AI ANALYSIS
# ============================================================

if analyze_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are an expert Data Analyst.

You are working with an e-commerce dataset.

Available columns:

{json.dumps(dataset_summary["columns"], indent=2)}

Numeric columns:

{json.dumps(dataset_summary["numeric_columns"], indent=2)}

Categorical columns:

{json.dumps(dataset_summary["categorical_columns"], indent=2)}

Your task is to understand the user's natural-language
question and convert it into a structured analysis instruction.

IMPORTANT:

Do NOT calculate the final numeric answer yourself.

Python/Pandas will perform the actual calculation.

Return ONLY valid JSON.

Use exactly:

{{
    "analysis_type": "single_value",
    "group_column": null,
    "value_column": "sales",
    "aggregation": "sum",
    "n": 10,
    "filter_year": null,
    "filter_category": null,
    "filter_region": null
}}

Allowed analysis_type:

single_value
group_by
top_n
bottom_n
negative_profit

Allowed aggregation:

sum
mean
count
min
max

Rules:

1. Sales means sales.

2. Revenue means sales.

3. Profit means profit.

4. Quantity means quantity.

5. Shipping cost means shipping_cost.

6. Discount means discount.

7. Orders means unique order_id.

8. "by category" means group_column = category.

9. "by sub-category" means group_column = sub_category.

10. "by region" means group_column = region.

11. "by segment" means group_column = segment.

12. "by market" means group_column = market.

13. "by state" means group_column = state.

14. "by country" means group_column = country.

15. "by product" means group_column = product_name.

16. "top 10 products by sales" means:
    analysis_type = top_n
    group_column = product_name
    value_column = sales
    aggregation = sum
    n = 10

17. "bottom 10 products by profit" means:
    analysis_type = bottom_n
    group_column = product_name
    value_column = profit
    aggregation = sum
    n = 10

18. "highest sales category" means:
    analysis_type = group_by
    group_column = category
    value_column = sales
    aggregation = sum

19. "highest profit region" means:
    analysis_type = group_by
    group_column = region
    value_column = profit
    aggregation = sum

20. If the user mentions a year such as 2014,
    set filter_year to 2014.

21. If the user asks for negative-profit products,
    use:
    analysis_type = negative_profit
    group_column = product_name
    value_column = profit
    aggregation = sum

22. Never invent column names.

23. Never return markdown.

24. Never return explanations.

25. Return JSON only.
"""


    # --------------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
User question:

{question}

Return only the JSON analysis instruction.
"""


    # --------------------------------------------------------
    # CALL HUGGING FACE
    # --------------------------------------------------------

    with st.spinner(
        "🤖 AI is understanding your question..."
    ):

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

                max_tokens=400,

                temperature=0.1
            )


            ai_text = (
                response
                .choices[0]
                .message.content
            )


            analysis = clean_json_response(
                ai_text
            )


            # ------------------------------------------------
            # JSON FALLBACK
            # ------------------------------------------------

            if analysis is None:

                st.warning(
                    "AI returned an unexpected response. "
                    "Trying again..."
                )

                response = client.chat.completions.create(

                    model=MODEL_NAME,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": (
                                user_prompt
                                + "\nRETURN JSON ONLY."
                            )
                        }
                    ],

                    max_tokens=400,

                    temperature=0
                )


                ai_text = (
                    response
                    .choices[0]
                    .message.content
                )


                analysis = clean_json_response(
                    ai_text
                )


            if analysis is None:

                st.error(
                    "The AI could not understand the question."
                )

                st.stop()


            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            analysis = normalize_analysis(
                analysis
            )


            # ------------------------------------------------
            # COPY DATA
            # ------------------------------------------------

            analysis_df = df.copy()


            # =================================================
            # YEAR FILTER
            # =================================================

            filter_year = analysis[
                "filter_year"
            ]

            if filter_year is not None:

                if "Year" in analysis_df.columns:

                    analysis_df = analysis_df[
                        analysis_df["Year"]
                        == filter_year
                    ]


            # =================================================
            # CATEGORY FILTER
            # =================================================

            filter_category = analysis[
                "filter_category"
            ]

            if (
                filter_category
                and "category" in analysis_df.columns
            ):

                analysis_df = analysis_df[
                    analysis_df["category"]
                    .astype(str)
                    .str.lower()
                    == str(filter_category).lower()
                ]


            # =================================================
            # REGION FILTER
            # =================================================

            filter_region = analysis[
                "filter_region"
            ]

            if (
                filter_region
                and "region" in analysis_df.columns
            ):

                analysis_df = analysis_df[
                    analysis_df["region"]
                    .astype(str)
                    .str.lower()
                    == str(filter_region).lower()
                ]


            # =================================================
            # CHECK EMPTY DATA
            # =================================================

            if analysis_df.empty:

                st.warning(
                    "No data found for the selected filters."
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


            # =================================================
            # SINGLE VALUE
            # =================================================

            if analysis_type == "single_value":

                st.subheader(
                    "🤖 AI Analysis"
                )


                if value_column is None:

                    st.error(
                        "I couldn't identify the metric."
                    )

                    st.stop()


                # --------------------------------------------
                # Orders
                # --------------------------------------------

                if value_column == "order_id":

                    result = analysis_df[
                        "order_id"
                    ].nunique()

                    label = "Unique Orders"


                else:

                    series = analysis_df[
                        value_column
                    ].dropna()


                    if aggregation == "sum":

                        result = series.sum()

                    elif aggregation == "mean":

                        result = series.mean()

                    elif aggregation == "min":

                        result = series.min()

                    elif aggregation == "max":

                        result = series.max()

                    elif aggregation == "count":

                        result = series.count()

                    else:

                        result = series.sum()


                    label = (
                        f"{aggregation.title()} "
                        f"{value_column.replace('_', ' ').title()}"
                    )


                st.success(
                    f"Based on your dataset, "
                    f"the {label.lower()} is "
                    f"**{result:,.2f}**."
                )


                st.metric(
                    label,
                    f"{result:,.2f}"
                )


            # =================================================
            # GROUP BY
            # =================================================

            elif analysis_type == "group_by":

                if (
                    group_column is None
                    or value_column is None
                ):

                    st.error(
                        "I couldn't determine how to group the data."
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


                highest = grouped.iloc[0]


                st.subheader(
                    "🤖 AI Analysis"
                )


                st.success(
                    f"**{highest[group_column]}** "
                    f"has the highest "
                    f"**{value_column.replace('_', ' ')}** "
                    f"with **{highest[value_column]:,.2f}**."
                )


                # --------------------------------------------
                # Chart
                # --------------------------------------------

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


            # =================================================
            # TOP N
            # =================================================

            elif analysis_type == "top_n":

                if (
                    group_column is None
                    or value_column is None
                ):

                    st.error(
                        "I couldn't determine the requested ranking."
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


                if grouped.empty:

                    st.warning(
                        "No results found."
                    )

                    st.stop()


                st.subheader(
                    f"🏆 Top {n} Results"
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


            # =================================================
            # BOTTOM N
            # =================================================

            elif analysis_type == "bottom_n":

                if (
                    group_column is None
                    or value_column is None
                ):

                    st.error(
                        "I couldn't determine the requested ranking."
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


                if grouped.empty:

                    st.warning(
                        "No results found."
                    )

                    st.stop()


                st.subheader(
                    f"⚠️ Bottom {n} Results"
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


            # =================================================
            # NEGATIVE PROFIT
            # =================================================

            elif analysis_type == "negative_profit":

                negative = (
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


                st.info(
                    f"{len(negative):,} "
                    f"products/groups have negative profit."
                )


                fig = px.bar(

                    negative.head(20),

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
                    negative,
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Something went wrong while analyzing the data."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Data Analyst | "
    "Python • Pandas • Streamlit • Hugging Face • Qwen3 • Plotly"
)
