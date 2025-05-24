import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("📊 Dynamic Data Visualization Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Car" in df.columns:
        df["Car"] = df["Car"].astype(str)

    st.write("### Preview of Data", df.head())

    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # Sidebar Filters
    st.sidebar.header("Filter Options")

    for col in ["SNAME", "GENDER", "CITY"]:
        if col in df.columns:
            selected_vals = st.sidebar.multiselect(f"Filter by {col}", options=df[col].unique())
            if selected_vals:
                df = df[df[col].isin(selected_vals)]

    # Optional Date Filter
    if 'DATE' in df.columns:
        try:
            df['DATE'] = pd.to_datetime(df['DATE'])
            use_date_filter = st.sidebar.checkbox("Filter by Date")
            if use_date_filter:
                start_date = st.sidebar.date_input("Start Date", df['DATE'].min().date())
                end_date = st.sidebar.date_input("End Date", df['DATE'].max().date())
                df = df[(df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)]
        except:
            st.sidebar.warning("'DATE' column could not be parsed to datetime.")

    st.sidebar.header("Choose Columns for Visualization")

    if categorical_columns and numeric_columns:
        x_axis = st.sidebar.selectbox("Select X-axis (categorical)", categorical_columns)
        y_axis = st.sidebar.selectbox("Select Y-axis (numeric)", numeric_columns)

        chart_type = st.sidebar.radio("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

        st.subheader(f"{chart_type} of {y_axis} by {x_axis}")

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Pie Chart":
            pie_data = df.groupby(x_axis)[y_axis].sum().reset_index()
            fig = px.pie(pie_data, values=y_axis, names=x_axis)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("The uploaded file must contain at least one numeric and one categorical column.")

    # 📈 Correlation Heatmap (Optional Toggle)
    st.subheader("📈 Correlation Heatmap (Numeric Columns Only)")
    if st.checkbox("Show Correlation Heatmap"):
        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns to display correlation heatmap.")

    # 🧠 Step 10: Basic Natural Language Summary of the Dataset
    st.subheader("🧠 AI Insights: Basic Data Summary")
    with st.expander("Show Summary"):
        buffer = io.StringIO()
        df.describe(include='all').to_string(buf=buffer)
        summary_text = buffer.getvalue()
        st.text(summary_text)

    # 📥 Step 11: Download Filtered Data as CSV
    st.subheader("📥 Download Filtered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

else:
    st.warning("Please upload a CSV file to get started.")
