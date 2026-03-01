
#!pip install streamlit
#!pip install pandas
#!pip install numpy
#!pip install scikit-learn


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
from streamlit_option_menu import option_menu

# ---------------------------
# Set Page Configuration
# ---------------------------
st.set_page_config(page_title="SFSG AI Tool", layout="wide")

# ---------------------------
# Constants
# ---------------------------
MODEL_PATH = "model.pkl"
TRAINING_DATA_PATH = "training_data.csv"

LABEL_OPTIONS = ["Ready", "Degraded", "Significant Degradation"]
STATUS_MAPPING = {"Ready": 0, "Degraded": 1, "Significant Degradation": 2}
STATUS_REVERSE = {v: k for k, v in STATUS_MAPPING.items()}

# ---------------------------
# Persistence Helpers
# ---------------------------
def load_model():
    """Returns the saved RandomForestClassifier or None if not yet trained."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def save_model(model):
    """Persists the model to MODEL_PATH."""
    joblib.dump(model, MODEL_PATH)


def load_training_data():
    """Loads accumulated training_data.csv, or returns an empty DataFrame."""
    if os.path.exists(TRAINING_DATA_PATH):
        return pd.read_csv(TRAINING_DATA_PATH)
    return pd.DataFrame(columns=["Sample_ID", "Quantity_Autosom_1", "Quantity_Autosom_2", "Degradation_Index", "Degradation_Status"])


def save_training_data(new_df):
    """Appends new_df to training_data.csv, upserting on Sample_ID."""
    existing = load_training_data()
    # Remove rows where Sample_ID already exists (so new data wins)
    existing = existing[~existing["Sample_ID"].isin(new_df["Sample_ID"])]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(TRAINING_DATA_PATH, index=False)
    return combined


# ---------------------------
# Utility Functions
# ---------------------------
def load_csv(file_path):
    """Loads a CSV file and handles errors."""
    try:
        df = pd.read_csv(file_path)
        st.write("Dataset loaded successfully:")
        st.write(df.head())

        # Validate Quantity column
        if 'Quantity' in df.columns:
            non_numeric_rows = df[pd.to_numeric(df['Quantity'], errors='coerce').isnull()]
            if not non_numeric_rows.empty:
                st.error("The 'Quantity' column contains non-numeric values. Please fix these rows:")
                st.write(non_numeric_rows)
                st.stop()
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


def calculate_mf_ratio(df):
    """Calculates the M:F ratio for the dataset."""
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Sample_ID'] = df['Sample_ID'].astype(str)
    df['Target_Name'] = df['Target_Name'].astype(str)

    male_data = df[df['Target_Name'] == 'Male']
    autosom2_data = df[df['Target_Name'] == 'Autosom 2']

    merged_data = pd.merge(male_data, autosom2_data, on='Sample_ID', suffixes=('_male', '_autosom2'))

    merged_data['Quantity_male'] = pd.to_numeric(merged_data['Quantity_male'], errors='coerce')
    merged_data['Quantity_autosom2'] = pd.to_numeric(merged_data['Quantity_autosom2'], errors='coerce')

    numerator = merged_data['Quantity_male'] / merged_data['Quantity_male']
    denominator = (merged_data['Quantity_autosom2'] - merged_data['Quantity_male']) / merged_data['Quantity_male']

    denominator = denominator.replace(0, float('nan'))

    merged_data['M:F_Ratio'] = numerator.fillna(0).astype(int).astype(str) + " : " + denominator.fillna(0).round(2).astype(str)

    final_data = merged_data[['Sample_ID', 'M:F_Ratio']]
    final_data.to_csv('output_with_MF_Ratio.csv', index=False)
    return final_data


def download_button(df, filename, label):
    """Creates a download button for a DataFrame."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
        type="primary"
    )


def save_csv(df, file_path):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
    st.success(f"Data saved to {file_path}")


def check_columns(df, required_columns):
    """Ensures required columns exist in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()


def classify_degradation(index):
    """Classify degradation based on the index."""
    if index < 1:
        return "Ready"
    elif 1 <= index < 10:
        return "Degraded"
    else:
        return "Significant Degradation"


def prepare_data(df):
    """Prepare dataset by cleaning and calculating the degradation index."""
    df.columns = df.columns.str.strip()

    required_columns = ['Sample_ID', 'Target_Name', 'Quantity']
    check_columns(df, required_columns)

    autosom_1 = df[df['Target_Name'] == 'Autosom 1'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_1'})
    autosom_2 = df[df['Target_Name'] == 'Autosom 2'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_2'})

    merged_df = pd.merge(autosom_1, autosom_2, on='Sample_ID', how='inner')

    merged_df['Quantity_Autosom_1'] = pd.to_numeric(merged_df['Quantity_Autosom_1'], errors='coerce')
    merged_df['Quantity_Autosom_2'] = pd.to_numeric(merged_df['Quantity_Autosom_2'], errors='coerce')

    merged_df['Degradation_Index'] = merged_df['Quantity_Autosom_2'] / merged_df['Quantity_Autosom_1']

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)

    invalid_rows = merged_df[merged_df.isnull().any(axis=1)]

    merged_df = merged_df.dropna()
    merged_df = merged_df.drop_duplicates(subset='Sample_ID', keep='first')

    return merged_df


# -----------------------
# Decision Making
# -----------------------

def assess_sample(df, sample_id):
    """Assess the degradation status of a specific sample."""
    sample_data = df[df['Sample_ID'] == sample_id]

    if sample_data.empty:
        st.error(f"Sample ID '{sample_id}' not found in the dataset.")
        return

    degradation_index = sample_data['Degradation_Index'].iloc[0]
    st.info(f" ##### The Degradation Index of the Sample ID {sample_id} is {degradation_index}")
    st.write(" ##### This Assessment tool calculates the Degradation Index by analyzing the ratio of specific target quantities in forensic samples, providing a quantitative measure of sample quality.\
    The implemented AI model leverages the calculated Degradation Index to classify samples into categories not degraded, Degraded, or Significant Degradation ensuring reliable assessment and actionable insights for forensic analysis.")

    model = load_model()

    if model is not None:
        features = sample_data[['Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index']].values
        prediction = model.predict(features)[0]
        label = STATUS_REVERSE[prediction]

        if label == "Ready":
            st.success(f"##### AI Assessment : The sample is not degraded and is ready for further analysis.")
        elif label == "Degraded":
            st.warning(f"##### AI Assessment : The sample is degraded. No action needed.")
        else:
            st.error(f"##### AI Assessment : The sample is significantly degraded. Resample is required.")
    else:
        st.info("No trained model found. Using rule-based thresholds — train a model first.")
        if degradation_index < 1:
            st.success("##### AI Assessment : The sample is not degraded and is ready for further analysis.")
        elif 1 <= degradation_index < 10:
            st.warning("##### AI Assessment : The sample is degraded. No action needed.")
        else:
            st.error("##### AI Assessment : The sample is significantly degraded. Resample is required.")


# ---------------------------
# Machine Learning Model
# ---------------------------

def train_model(labeled_df):
    """Train and evaluate a RandomForestClassifier using analyst-reviewed labels."""
    # labeled_df must have 'Degradation_Status' as string labels
    train_df = labeled_df[['Sample_ID', 'Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index', 'Degradation_Status']].copy()

    # Map string labels to integers
    train_df['Degradation_Status'] = train_df['Degradation_Status'].map(STATUS_MAPPING)

    # Accumulate: upsert into training_data.csv
    accumulated = save_training_data(train_df)

    total_samples = len(accumulated)
    st.write(f"### Training on {total_samples} accumulated samples")

    X = accumulated[['Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index']]
    y = accumulated['Degradation_Status'].astype(int)

    if X.isnull().values.any() or np.isinf(X.values).any():
        st.error("Accumulated training data contains null or infinite values. Please reset and re-upload a clean dataset.")
        return

    if len(accumulated) < 5:
        st.warning("Very few samples — model will train but may not generalise well. Upload more data to improve accuracy.")

    # Use train/test split only when enough samples exist
    if len(accumulated) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("### Model Evaluation")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.info("Fewer than 10 samples — skipping train/test split. Evaluation will be available once more data is accumulated.")

    save_model(model)
    st.success(f"Model trained and saved. Total training samples: {total_samples}")


# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        img = Image.open("SFSG_Logo.png")
        st.image(img, width=300)
        app = option_menu(
            menu_title='Main Menu',
            options=['About Us', "AI Sample Assessment Tool"],
            icons=['house-fill', 'person-circle'],
            menu_icon='chat-text-fill',
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": 'green'},
                "icon": {"color": "white", "font-size": "20px"},
                "menu_title": {"background-color": "white"},
                "nav-link": {"color": "white", "font-size": "20px", "text-align": "center", "margin": "1px", "--hover-color": "#8D272B"},
                "nav-link-selected": {"background-color": "green"},
            }
        )

        # Model status in sidebar
        st.divider()
        model = load_model()
        if model is not None:
            training_data = load_training_data()
            st.success(f"Model trained on {len(training_data)} samples")
        else:
            st.warning("No model trained yet")

        # Reset model button
        if st.button("Reset Model", type="secondary"):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(TRAINING_DATA_PATH):
                os.remove(TRAINING_DATA_PATH)
            st.success("Model and training data cleared.")
            st.rerun()

    if app == "About Us":
        st.title("About Science For Social Good CIC")
        st.info("\n Vocational training transforms lives and drives economic growth in developing countries.\
                   It equips individuals with the practical skills and knowledge needed to secure employment, start businesses, and contribute to their community's development.\
                   Through our work, we constantly explore the significance of vocational training in developing countries and its positive impact on individuals, communities, and overall socio-economic progress.\
                   \n Our direction is genetic analysis testing verticals in Agriculture, Human Identification (Forensic DNA), quality and safety testing (environmental, food and water).\
                   To achieve this goal, SSG-CIC aims to establish Vocational Training Hubs (VTHs) across the globe and build local capacity for training and development.\
                   All of this is possible with your support and volunteering.  \n How you can support us?\
                   \n Our fundraising is through 4 main routes:\
                       \n 1. Non Profit Consulting\
                        \n 2. Crowdfunding\
                        \n 3. Volunteer and Work with us\
                        \n 4. Donate used laboratory equipment, IT Gear and Software")

        st.link_button("\n Click here to view our website", "https://ssg-cic.org/", type="primary")

    if app == "AI Sample Assessment Tool":
        st.title("🤖AI Assessment Tool for Forensic Sample Analysis")

        st.info(" \n The AI-powered Assessment Tool is a comprehensive and automated solution for forensic sample quality assessment.\
                    By combining structured decision-making with advanced machine learning techniques, it Saves time and improves efficiency in forensic workflows, provides accurate, reliable assessments.\
                    It also Enhances decision-making with clear, actionable insights.\
                    This tool is an invaluable asset for forensic labs, ensuring quality control and minimizing errors in sample analysis")

        uploaded_file = st.file_uploader(" Upload the sample dataset here in CSV Format", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### The Sample Data is uploaded Successfully")
            st.write(df.head())
            prepared_data = prepare_data(df)

            # ---------------------------
            # Label Editing UI
            # ---------------------------
            st.subheader("Review and Edit Sample Labels")
            st.write("Auto-generated labels are shown below. You can correct any label before training the model.")

            label_df = prepared_data.copy()
            label_df['Label'] = label_df['Degradation_Index'].apply(classify_degradation)

            edited = st.data_editor(
                label_df[['Sample_ID', 'Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index', 'Label']],
                column_config={
                    "Label": st.column_config.SelectboxColumn(
                        "Label",
                        options=LABEL_OPTIONS,
                        required=True,
                    )
                },
                use_container_width=True,
                hide_index=True,
                key=f"label_editor_{uploaded_file.name}_{uploaded_file.size}"
            )

            # ---------------------------
            # Train Model
            # ---------------------------
            st.subheader("Train the AI Model")
            if st.button("Click here to Train Model", type="primary"):
                train_df = edited.rename(columns={"Label": "Degradation_Status"})
                train_model(train_df)

            # ---------------------------
            # M:F Ratio
            # ---------------------------
            if st.button("Click here to Calculate M:F Ratio for the uploaded dataset", type="primary"):
                df_with_ratio = calculate_mf_ratio(df)
                st.write("### Dataset with M:F Ratio")
                st.write(df_with_ratio.head())
                download_button(df_with_ratio, "Dataset_with_MF_Ratio.csv", "Download Dataset with M:F Ratio")

            # ---------------------------
            # Assess Sample
            # ---------------------------
            st.subheader("🔍 Assess Sample")
            sample_id = st.text_input(" ##### Enter the Sample ID here")
            if sample_id:
                assess_sample(prepared_data, sample_id)


if __name__ == "__main__":
    main()
