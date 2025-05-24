<<<<<<< HEAD

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Check for statsmodels
try:
    import statsmodels
    trendline_available = True
except ImportError:
    trendline_available = False

st.set_page_config(layout="wide")
st.title("📊 Dynamic Data Visualization Dashboard")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

@st.cache_data
def clean_data(df, fill_method='mean', custom_fill_value=None, interpolate_method=None, 
               drop_duplicates=True, remove_outliers=False, drop_high_null_rows=False, 
               null_threshold=0.5, replace_values=None, sample_size=None):
    df_cleaned = df.copy()
    
    # Handle missing values
    with st.spinner("Handling missing values..."):
        if fill_method == 'custom' and custom_fill_value is not None:
            if isinstance(custom_fill_value, (int, float, str)):
                df_cleaned = df_cleaned.fillna(custom_fill_value)
            else:
                st.warning("Custom fill value must be a number or string.")
        elif fill_method == 'ffill':
            df_cleaned = df_cleaned.ffill()
        elif fill_method == 'bfill':
            df_cleaned = df_cleaned.bfill()
        elif fill_method == 'interpolate' and interpolate_method:
            df_cleaned = df_cleaned.interpolate(method=interpolate_method)
        elif fill_method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
        elif fill_method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
        elif fill_method == 'mode':
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                mode = df_cleaned[col].mode()
                if not mode.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode[0])
        elif fill_method == 'drop':
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            if len(df_cleaned) < initial_rows:
                st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with missing values.")

    # Drop rows with high percentage of null values
    if drop_high_null_rows:
        initial_rows = len(df_cleaned)
        threshold = len(df_cleaned.columns) * null_threshold
        df_cleaned = df_cleaned.dropna(thresh=threshold)
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with more than {int(null_threshold*100)}% null values.")

    # Drop duplicates
    if drop_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} duplicate rows.")

    # Remove outliers (using IQR method)
    if remove_outliers:
        numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            if len(df_cleaned) < initial_rows:
                st.warning(f"Removed {initial_rows - len(df_cleaned)} rows with outliers in {col}.")

    # Replace specific values
    if replace_values:
        for old_value, new_value in replace_values.items():
            df_cleaned = df_cleaned.replace(old_value, new_value)
            st.warning(f"Replaced '{old_value}' with '{new_value}'.")

    # Apply sampling if specified
    if sample_size and len(df_cleaned) > sample_size:
        st.warning(f"Dataset is large. Sampling to {sample_size} rows for performance.")
        df_cleaned = df_cleaned.sample(n=sample_size, random_state=42)

    return df_cleaned

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Tabs for Data Cleaning, Visualization, and Insights
    tab1, tab2, tab3 = st.tabs(["🧹 Clean Data", "💻 Filters & Visualization", "🧠 AI Insights"])

    with tab1:
        # Sampling Option
        use_sampling = st.checkbox("Sample Large Datasets", value=len(df) > 10000)
        sample_size = st.slider("Sample Size (rows)", 1000, min(100000, len(df)), 10000, step=1000) if use_sampling else None

        # Clean Data Options
        with st.expander("Data Cleaning Options"):
            fill_method = st.selectbox("Handle Missing Values", ["mean", "median", "mode", "custom", "ffill", "bfill", "interpolate", "drop"])
            if fill_method == "custom":
                custom_fill_value = st.text_input("Enter custom fill value (number or string)")
                try:
                    custom_fill_value = float(custom_fill_value) if custom_fill_value.replace('.','',1).isdigit() else custom_fill_value
                except ValueError:
                    custom_fill_value = custom_fill_value
            else:
                custom_fill_value = None
            if fill_method == "interpolate":
                interpolate_method = st.selectbox("Interpolation Method", ["linear", "quadratic", "cubic"])
            else:
                interpolate_method = None
            drop_duplicates = st.checkbox("Drop Duplicate Rows", value=True)
            remove_outliers = st.checkbox("Remove Outliers (IQR Method)", value=False)
            drop_high_null_rows = st.checkbox("Drop Rows with High Null Percentage", value=False)
            null_threshold = st.slider("Null Threshold (%)", 0.0, 1.0, 0.5, 0.1) if drop_high_null_rows else 0.5
            replace_values = {}
            if st.checkbox("Replace Specific Values"):
                cols = df.columns.tolist()
                col_to_replace = st.selectbox("Select Column to Replace Values", cols)
                old_value = st.text_input("Old Value to Replace")
                new_value = st.text_input("New Value")
                if old_value and new_value:
                    replace_values[old_value] = new_value
            if st.button("Apply Cleaning"):
                with st.spinner("Cleaning data..."):
                    df = clean_data(df, fill_method, custom_fill_value, interpolate_method, drop_duplicates, 
                                   remove_outliers, drop_high_null_rows, null_threshold, replace_values, sample_size)
                    st.success("Data cleaned successfully!")

    with tab2:
        # Data Preview
        st.write("### Preview of Data", df.head())
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        # Sidebar Filters
        st.sidebar.header("Filter Options")
        for col in categorical_columns:
            selected_vals = st.sidebar.multiselect(f"Filter by {col}", options=df[col].unique())
            if selected_vals:
                df = df[df[col].isin(selected_vals)]

        # Date Filter
        if date_columns:
            date_col = st.sidebar.selectbox("Select Date Column", date_columns)
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                if df[date_col].isna().all():
                    st.sidebar.warning(f"Column '{date_col}' contains no valid dates.")
                else:
                    use_date_filter = st.sidebar.checkbox("Filter by Date")
                    if use_date_filter:
                        start_date = st.sidebar.date_input("Start Date", df[date_col].min().date())
                        end_date = st.sidebar.date_input("End Date", df[date_col].max().date())
                        df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
            except Exception as e:
                st.sidebar.warning(f"Could not parse '{date_col}' as datetime: {str(e)}")

        # Custom Query Filter
        st.subheader("💻 Custom Query Filter")
        with st.expander("Write Custom Query to Filter Data"):
            query = st.text_area("Enter query (e.g., Glucose > 100 and Age < 50)", height=100)
            if st.button("Apply Custom Query"):
                try:
                    df = df.query(query)
                    st.success("Custom query applied successfully!")
                except Exception as e:
                    st.error(f"Invalid query: {str(e)}")

        # Visualization
        st.sidebar.header("Visualization Options")
        if categorical_columns and numeric_columns:
            x_axis = st.sidebar.selectbox("Select X-axis (categorical)", categorical_columns)
            y_axis = st.sidebar.multiselect("Select Y-axis (numeric)", numeric_columns, default=[numeric_columns[0]] if numeric_columns else [])
            chart_type = st.sidebar.radio("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Violin Plot"])
            show_trendline = st.sidebar.checkbox("Show Trendline (Scatter Plot)", value=False, disabled=not trendline_available) if chart_type == "Scatter Plot" else False

            st.subheader(f"{chart_type} of {', '.join(y_axis)} by {x_axis}")
            if not y_axis:
                st.warning("Please select at least one Y-axis column.")
            else:
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Pie Chart":
                    agg_method = st.sidebar.selectbox("Aggregation Method", ["sum", "mean", "count"])
                    if agg_method == "sum":
                        pie_data = df.groupby(x_axis)[y_axis[0]].sum().reset_index()
                    elif agg_method == "mean":
                        pie_data = df.groupby(x_axis)[y_axis[0]].mean().reset_index()
                    else:
                        pie_data = df.groupby(x_axis)[y_axis[0]].count().reset_index()
                        pie_data.columns = [x_axis, y_axis[0]]
                    fig = px.pie(pie_data, values=y_axis[0], names=x_axis)
                elif chart_type == "Scatter Plot":
                    if show_trendline and trendline_available:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], color=x_axis, trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis[0], color=x_axis, box=True)
                st.plotly_chart(fig, use_container_width=True)
        elif numeric_columns and not categorical_columns:
            x_axis = st.sidebar.selectbox("Select X-axis (numeric)", numeric_columns)
            y_axis = st.sidebar.multiselect("Select Y-axis (numeric)", numeric_columns, default=[numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]])
            chart_type = st.sidebar.radio("Chart Type", ["Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Violin Plot"])
            show_trendline = st.sidebar.checkbox("Show Trendline (Scatter Plot)", value=False, disabled=not trendline_available) if chart_type == "Scatter Plot" else False

            st.subheader(f"{chart_type} of {', '.join(y_axis)} vs {x_axis}")
            if not y_axis:
                st.warning("Please select at least one Y-axis column.")
            else:
                if chart_type == "Scatter Plot":
                    if show_trendline and trendline_available:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0])
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, nbins=20)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis[0])
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis[0], box=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("The uploaded file must contain at least one numeric column for visualization.")

        # Correlation Heatmap
        st.subheader("📈 Correlation Heatmap")
        if len(numeric_columns) < 2:
            st.warning("Not enough numeric columns to display correlation heatmap.")
        elif st.checkbox("Show Correlation Heatmap"):
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    with tab3:
        # AI Insights
        st.subheader("🧠 AI Insights: Data Summary")
        with st.expander("Show Summary"):
            st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.write("### Numeric Columns")
            for col in numeric_columns:
                mean = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                skewness = df[col].skew()
                st.write(f"{col}: Mean = {mean:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}, Skewness = {skewness:.2f}")
                if abs(skewness) > 1:
                    st.write(f"⚠️ {col} is highly skewed. Consider transformation (e.g., log).")
            st.write("### Categorical Columns")
            for col in categorical_columns:
                st.write(f"{col}: {df[col].nunique()} unique values")
            if len(numeric_columns) >= 2:
                st.write("### Key Correlations")
                corr = df[numeric_columns].corr()
                high_corr = corr[(corr > 0.7) | (corr < -0.7)]
                for col1 in high_corr.columns:
                    for col2 in high_corr.index:
                        if col1 < col2 and not pd.isna(high_corr.loc[col2, col1]) and abs(high_corr.loc[col2, col1]) > 0.7:
                            st.write(f"Strong correlation between {col1} and {col2}: {high_corr.loc[col2, col1]:.2f}")

    # Download Filtered Data
    st.subheader("📥 Download Filtered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
else:
=======

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Check for statsmodels
try:
    import statsmodels
    trendline_available = True
except ImportError:
    trendline_available = False

st.set_page_config(layout="wide")
st.title("📊 Dynamic Data Visualization Dashboard")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

@st.cache_data
def clean_data(df, fill_method='mean', custom_fill_value=None, interpolate_method=None, 
               drop_duplicates=True, remove_outliers=False, drop_high_null_rows=False, 
               null_threshold=0.5, replace_values=None, sample_size=None):
    df_cleaned = df.copy()
    
    # Handle missing values
    with st.spinner("Handling missing values..."):
        if fill_method == 'custom' and custom_fill_value is not None:
            if isinstance(custom_fill_value, (int, float, str)):
                df_cleaned = df_cleaned.fillna(custom_fill_value)
            else:
                st.warning("Custom fill value must be a number or string.")
        elif fill_method == 'ffill':
            df_cleaned = df_cleaned.ffill()
        elif fill_method == 'bfill':
            df_cleaned = df_cleaned.bfill()
        elif fill_method == 'interpolate' and interpolate_method:
            df_cleaned = df_cleaned.interpolate(method=interpolate_method)
        elif fill_method == 'mean':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
        elif fill_method == 'median':
            numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
        elif fill_method == 'mode':
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                mode = df_cleaned[col].mode()
                if not mode.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode[0])
        elif fill_method == 'drop':
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            if len(df_cleaned) < initial_rows:
                st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with missing values.")

    # Drop rows with high percentage of null values
    if drop_high_null_rows:
        initial_rows = len(df_cleaned)
        threshold = len(df_cleaned.columns) * null_threshold
        df_cleaned = df_cleaned.dropna(thresh=threshold)
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} rows with more than {int(null_threshold*100)}% null values.")

    # Drop duplicates
    if drop_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(df_cleaned)} duplicate rows.")

    # Remove outliers (using IQR method)
    if remove_outliers:
        numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            if len(df_cleaned) < initial_rows:
                st.warning(f"Removed {initial_rows - len(df_cleaned)} rows with outliers in {col}.")

    # Replace specific values
    if replace_values:
        for old_value, new_value in replace_values.items():
            df_cleaned = df_cleaned.replace(old_value, new_value)
            st.warning(f"Replaced '{old_value}' with '{new_value}'.")

    # Apply sampling if specified
    if sample_size and len(df_cleaned) > sample_size:
        st.warning(f"Dataset is large. Sampling to {sample_size} rows for performance.")
        df_cleaned = df_cleaned.sample(n=sample_size, random_state=42)

    return df_cleaned

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Tabs for Data Cleaning, Visualization, and Insights
    tab1, tab2, tab3 = st.tabs(["🧹 Clean Data", "💻 Filters & Visualization", "🧠 AI Insights"])

    with tab1:
        # Sampling Option
        use_sampling = st.checkbox("Sample Large Datasets", value=len(df) > 10000)
        sample_size = st.slider("Sample Size (rows)", 1000, min(100000, len(df)), 10000, step=1000) if use_sampling else None

        # Clean Data Options
        with st.expander("Data Cleaning Options"):
            fill_method = st.selectbox("Handle Missing Values", ["mean", "median", "mode", "custom", "ffill", "bfill", "interpolate", "drop"])
            if fill_method == "custom":
                custom_fill_value = st.text_input("Enter custom fill value (number or string)")
                try:
                    custom_fill_value = float(custom_fill_value) if custom_fill_value.replace('.','',1).isdigit() else custom_fill_value
                except ValueError:
                    custom_fill_value = custom_fill_value
            else:
                custom_fill_value = None
            if fill_method == "interpolate":
                interpolate_method = st.selectbox("Interpolation Method", ["linear", "quadratic", "cubic"])
            else:
                interpolate_method = None
            drop_duplicates = st.checkbox("Drop Duplicate Rows", value=True)
            remove_outliers = st.checkbox("Remove Outliers (IQR Method)", value=False)
            drop_high_null_rows = st.checkbox("Drop Rows with High Null Percentage", value=False)
            null_threshold = st.slider("Null Threshold (%)", 0.0, 1.0, 0.5, 0.1) if drop_high_null_rows else 0.5
            replace_values = {}
            if st.checkbox("Replace Specific Values"):
                cols = df.columns.tolist()
                col_to_replace = st.selectbox("Select Column to Replace Values", cols)
                old_value = st.text_input("Old Value to Replace")
                new_value = st.text_input("New Value")
                if old_value and new_value:
                    replace_values[old_value] = new_value
            if st.button("Apply Cleaning"):
                with st.spinner("Cleaning data..."):
                    df = clean_data(df, fill_method, custom_fill_value, interpolate_method, drop_duplicates, 
                                   remove_outliers, drop_high_null_rows, null_threshold, replace_values, sample_size)
                    st.success("Data cleaned successfully!")

    with tab2:
        # Data Preview
        st.write("### Preview of Data", df.head())
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        # Sidebar Filters
        st.sidebar.header("Filter Options")
        for col in categorical_columns:
            selected_vals = st.sidebar.multiselect(f"Filter by {col}", options=df[col].unique())
            if selected_vals:
                df = df[df[col].isin(selected_vals)]

        # Date Filter
        if date_columns:
            date_col = st.sidebar.selectbox("Select Date Column", date_columns)
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                if df[date_col].isna().all():
                    st.sidebar.warning(f"Column '{date_col}' contains no valid dates.")
                else:
                    use_date_filter = st.sidebar.checkbox("Filter by Date")
                    if use_date_filter:
                        start_date = st.sidebar.date_input("Start Date", df[date_col].min().date())
                        end_date = st.sidebar.date_input("End Date", df[date_col].max().date())
                        df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
            except Exception as e:
                st.sidebar.warning(f"Could not parse '{date_col}' as datetime: {str(e)}")

        # Custom Query Filter
        st.subheader("💻 Custom Query Filter")
        with st.expander("Write Custom Query to Filter Data"):
            query = st.text_area("Enter query (e.g., Glucose > 100 and Age < 50)", height=100)
            if st.button("Apply Custom Query"):
                try:
                    df = df.query(query)
                    st.success("Custom query applied successfully!")
                except Exception as e:
                    st.error(f"Invalid query: {str(e)}")

        # Visualization
        st.sidebar.header("Visualization Options")
        if categorical_columns and numeric_columns:
            x_axis = st.sidebar.selectbox("Select X-axis (categorical)", categorical_columns)
            y_axis = st.sidebar.multiselect("Select Y-axis (numeric)", numeric_columns, default=[numeric_columns[0]] if numeric_columns else [])
            chart_type = st.sidebar.radio("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Box Plot", "Violin Plot"])
            show_trendline = st.sidebar.checkbox("Show Trendline (Scatter Plot)", value=False, disabled=not trendline_available) if chart_type == "Scatter Plot" else False

            st.subheader(f"{chart_type} of {', '.join(y_axis)} by {x_axis}")
            if not y_axis:
                st.warning("Please select at least one Y-axis column.")
            else:
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Pie Chart":
                    agg_method = st.sidebar.selectbox("Aggregation Method", ["sum", "mean", "count"])
                    if agg_method == "sum":
                        pie_data = df.groupby(x_axis)[y_axis[0]].sum().reset_index()
                    elif agg_method == "mean":
                        pie_data = df.groupby(x_axis)[y_axis[0]].mean().reset_index()
                    else:
                        pie_data = df.groupby(x_axis)[y_axis[0]].count().reset_index()
                        pie_data.columns = [x_axis, y_axis[0]]
                    fig = px.pie(pie_data, values=y_axis[0], names=x_axis)
                elif chart_type == "Scatter Plot":
                    if show_trendline and trendline_available:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], color=x_axis, trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis[0], color=x_axis)
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis[0], color=x_axis, box=True)
                st.plotly_chart(fig, use_container_width=True)
        elif numeric_columns and not categorical_columns:
            x_axis = st.sidebar.selectbox("Select X-axis (numeric)", numeric_columns)
            y_axis = st.sidebar.multiselect("Select Y-axis (numeric)", numeric_columns, default=[numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]])
            chart_type = st.sidebar.radio("Chart Type", ["Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Violin Plot"])
            show_trendline = st.sidebar.checkbox("Show Trendline (Scatter Plot)", value=False, disabled=not trendline_available) if chart_type == "Scatter Plot" else False

            st.subheader(f"{chart_type} of {', '.join(y_axis)} vs {x_axis}")
            if not y_axis:
                st.warning("Please select at least one Y-axis column.")
            else:
                if chart_type == "Scatter Plot":
                    if show_trendline and trendline_available:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0], trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis[0])
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, nbins=20)
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis[0])
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis[0], box=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("The uploaded file must contain at least one numeric column for visualization.")

        # Correlation Heatmap
        st.subheader("📈 Correlation Heatmap")
        if len(numeric_columns) < 2:
            st.warning("Not enough numeric columns to display correlation heatmap.")
        elif st.checkbox("Show Correlation Heatmap"):
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

    with tab3:
        # AI Insights
        st.subheader("🧠 AI Insights: Data Summary")
        with st.expander("Show Summary"):
            st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.write("### Numeric Columns")
            for col in numeric_columns:
                mean = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                skewness = df[col].skew()
                st.write(f"{col}: Mean = {mean:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}, Skewness = {skewness:.2f}")
                if abs(skewness) > 1:
                    st.write(f"⚠️ {col} is highly skewed. Consider transformation (e.g., log).")
            st.write("### Categorical Columns")
            for col in categorical_columns:
                st.write(f"{col}: {df[col].nunique()} unique values")
            if len(numeric_columns) >= 2:
                st.write("### Key Correlations")
                corr = df[numeric_columns].corr()
                high_corr = corr[(corr > 0.7) | (corr < -0.7)]
                for col1 in high_corr.columns:
                    for col2 in high_corr.index:
                        if col1 < col2 and not pd.isna(high_corr.loc[col2, col1]) and abs(high_corr.loc[col2, col1]) > 0.7:
                            st.write(f"Strong correlation between {col1} and {col2}: {high_corr.loc[col2, col1]:.2f}")

    # Download Filtered Data
    st.subheader("📥 Download Filtered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
else:
>>>>>>> 51111110d8abea7c9109ed8a02ff99e67be949f1
    st.warning("Please upload a CSV file to get started.")