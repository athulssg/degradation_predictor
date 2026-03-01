
#!pip install streamlit
#!pip install pandas
#!pip install numpy
#!pip install scikit-learn


import streamlit as st
import pandas as pd
import numpy as np
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
    # Assuming 'df' is your input DataFrame
    # Convert 'Quantity' columns to numeric, coercing errors to NaN
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    # Ensure 'Sample_ID' and 'Target_Name' columns are strings (if required)
    df['Sample_ID'] = df['Sample_ID'].astype(str)
    df['Target_Name'] = df['Target_Name'].astype(str)

    # Extract rows for 'Male' and 'Autosom 2' based on 'Target_Name'
    male_data = df[df['Target_Name'] == 'Male']
    autosom2_data = df[df['Target_Name'] == 'Autosom 2']

    # Merge male and autosom2 data on 'Sample_ID' to align them correctly
    merged_data = pd.merge(male_data, autosom2_data, on='Sample_ID', suffixes=('_male', '_autosom2'))

    # Calculate the components for the M:F ratio
    # Ensure the merged columns 'Quantity_male' and 'Quantity_autosom2' are numeric
    merged_data['Quantity_male'] = pd.to_numeric(merged_data['Quantity_male'], errors='coerce')
    merged_data['Quantity_autosom2'] = pd.to_numeric(merged_data['Quantity_autosom2'], errors='coerce')

    # Numerator formula
    numerator = merged_data['Quantity_male'] / merged_data['Quantity_male']
    # Denominator formula
    denominator = (merged_data['Quantity_autosom2'] - merged_data['Quantity_male']) / merged_data['Quantity_male']

    # Handle cases where denominator might be zero or numerator has NaN
    denominator = denominator.replace(0, float('nan'))

    # Format M:F Ratio as "A : B"
    merged_data['M:F_Ratio'] = numerator.fillna(0).astype(int).astype(str) + " : " + denominator.fillna(0).round(2).astype(str)

    # Display the output
    #print(merged_data[['Sample_ID', 'M:F_Ratio']])
    final_data = merged_data[['Sample_ID', 'M:F_Ratio']]
    # Save the result to a CSV file
    final_data.to_csv('output_with_MF_Ratio.csv', index=False)
    #df['M:F Ratio'] = df['Quantity'] / (df['Quantity'].mean())
    return final_data

def download_button(df, filename, label):
    """Creates a download button for a DataFrame."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
        type= "primary"
    )

def save_csv(df, file_path):
    """Saves a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
    st.success(f"Data saved to {file_path}")


#def load_csv(file_path):
    #Loads a CSV file and handles errors."""
    #try:
        #df = pd.read_csv(file_path)
        #st.write("Dataset loaded successfully:")
        #st.write(df.head())
        #return df
    #except FileNotFoundError:
        #st.error(f"Error: '{file_path}' not found.")
    #except Exception as e:
        #st.error(f"An unexpected error occurred: {e}")"""
        

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

    # Filter Autosom 1 and Autosom 2 data
    autosom_1 = df[df['Target_Name'] == 'Autosom 1'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_1'})
    autosom_2 = df[df['Target_Name'] == 'Autosom 2'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_2'})

    # Merge datasets on Sample_ID
    merged_df = pd.merge(autosom_1, autosom_2, on='Sample_ID', how='inner')

    # Convert quantities to numeric and handle errors
    merged_df['Quantity_Autosom_1'] = pd.to_numeric(merged_df['Quantity_Autosom_1'], errors='coerce')
    merged_df['Quantity_Autosom_2'] = pd.to_numeric(merged_df['Quantity_Autosom_2'], errors='coerce')

    # Calculate the Degradation Index
    merged_df['Degradation_Index'] = merged_df['Quantity_Autosom_2'] / merged_df['Quantity_Autosom_1']

    # Replace inf/-inf with NaN
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)

    # Log problematic rows for debugging
    invalid_rows = merged_df[merged_df.isnull().any(axis=1)]
    #if not invalid_rows.empty:
        #st.write("### Invalid Rows:")
        #st.write(invalid_rows)

    # Drop rows with NaN values
    merged_df = merged_df.dropna()

    # Drop duplicate entries based on Sample_ID
    merged_df = merged_df.drop_duplicates(subset='Sample_ID', keep='first')

    return merged_df

# ---------------------------
# Data Preparation
# ---------------------------
#def prepare_data(df):
    #Prepare dataset by cleaning and calculating the degradation index.
    #df.columns = df.columns.str.strip()

    #required_columns = ['Sample_ID', 'Target_Name', 'Quantity']
    #check_columns(df, required_columns)

    #autosom_1 = df[df['Target_Name'] == 'Autosom 1'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_1'})
    #autosom_2 = df[df['Target_Name'] == 'Autosom 2'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_2'})

    #merged_df = pd.merge(autosom_1, autosom_2, on='Sample_ID', how='inner')
    #merged_df['Quantity_Autosom_1'] = pd.to_numeric(merged_df['Quantity_Autosom_1'], errors='coerce')
    #merged_df['Quantity_Autosom_2'] = pd.to_numeric(merged_df['Quantity_Autosom_2'], errors='coerce')

    #merged_df['Degradation_Index'] = merged_df['Quantity_Autosom_2'] / merged_df['Quantity_Autosom_1']
    #merged_df = merged_df.drop_duplicates(subset='Sample_ID', keep='first')

    #return merged_df"""

# -----------------------
# Decision Making
# -----------------------

def assess_sample(df, sample_id):
    """Assess the degradation status of a specific sample."""
    sample_data = df[df['Sample_ID'] == sample_id]

    if sample_data.empty:
        st.error(f"Sample ID '{sample_id}' not found in the dataset.")
    else:
        degradation_index = sample_data['Degradation_Index'].iloc[0]
        #st.write(f"Sample ID: {sample_id}")
        st.info(f" ##### The Degradation Index of the Sample ID {sample_id} is {degradation_index}")
        st.write(" ##### This Assesement tool calculates the Degradation Index by analyzing the ratio of specific target quantities in forensic samples, providing a quantitative measure of sample quality.\
        The implemented AI model leverages the calculated Degradation Index to classify samples into categories not degraded, Degraded, or Significant Degradation ensuring reliable assessment and actionable insights for forensic analysis.")
        

        if degradation_index < 1:
            st.success("##### AI Assesement : The sample is not degraded and is ready for further analysis.")
        elif 1 <= degradation_index < 10:
            st.warning("##### AI Assesement : The sample is degraded. No action needed.")
        else:
            st.error("##### AI Assesement : The sample is significantly degraded. Resample is required.")
            
# ---------------------------
# Machine Learning Model
# ---------------------------

def train_model(df):
    """Train and evaluate a RandomForestClassifier."""
    df['Degradation_Status'] = df['Degradation_Index'].apply(classify_degradation)
    status_mapping = {"Ready": 0, "Degraded": 1, "Significant Degradation": 2}
    df['Degradation_Status'] = df['Degradation_Status'].map(status_mapping)

    X = df[['Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index']]
    y = df['Degradation_Status']

    if X.isnull().values.any() or np.isinf(X.values).any():
        st.error("Input data contains null or infinite values. Please ensure the dataset is clean before training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### Model Evaluation")
    st.text(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#def train_model(df):
    #Train and evaluate a RandomForestClassifier.
    #df['Degradation_Status'] = df['Degradation_Index'].apply(classify_degradation)
    #status_mapping = {"Ready": 0, "Degraded": 1, "Significant Degradation": 2}
   # df['Degradation_Status'] = df['Degradation_Status'].map(status_mapping)

   # X = df[['Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index']]
   # y = df['Degradation_Status']

   #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #model = RandomForestClassifier(random_state=42)
   # model.fit(X_train, y_train)

    #y_pred = model.predict(X_test)

    #st.write("### Model Evaluation")
    #st.text(classification_report(y_test, y_pred))
   # st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}") 


# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
   
    #st.set_page_config(page_title = "SFSG AI Tool", layout="wide")
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    #df = pd.read_csv("SFSG_Dataset.csv")
    with st.sidebar:   
            img = Image.open( "SFSG_Logo.png")
            st.image(img, width =300)
            app = option_menu(
                menu_title='Main Menu',
                options=['About Us', "AI Sample Assesement Tool"],
                icons=['house-fill','person-circle'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "0!important","background-color":'green'},
                    "icon": {"color": "white", "font-size": "20px"}, 
                    "menu_title":{"background-color": "white"} ,    
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "center", "margin":"1px", "--hover-color": "#8D272B"},
                    "nav-link-selected": {"background-color": "green"},}
            )
        
    if app =="About Us":
            st.title("About Science For Social Good CIC")
            st.info ("\n Vocational training transforms lives and drives economic growth in developing countries.\
                       It equips individuals with the practical skills and knowledge needed to secure employment, start businesses, and contribute to their community’s development.\
                       Through our work, we constantly explore the significance of vocational training in developing countries and its positive impact on individuals, communities, and overall socio-economic progress.\
                       \n Our direction is genetic analysis testing verticals in Agriculture, Human Identification (Forensic DNA), quality and safety testing (environmental, food and water).\
                       To achieve this goal, SSG-CIC aims to establish Vocational Training Hubs (VTHs) across the globe and build local capacity for training and development.\
                       All of this is possible with your support and volunteering.  \n How you can support us?\
                       \n Our fundraising is through 4 main routes:\
                           \n 1. Non Profit Consulting\
                            \n 2. Crowdfunding\
                            \n 3. Volunteer and Work with us\
                            \n 4. Donate used laboratory equipment, IT Gear and Software")

            st.link_button("\n Click here to view our website", "https://ssg-cic.org/", type = "primary")
        
    if app == "AI Sample Assesement Tool":
            st.title("🤖AI Assesement Tool for Forensic Sample Analysis")
            
            st.info(" \n The AI-powered Assesement Tool is a comprehensive and automated solution for forensic sample quality assessment.\
                        By combining structured decision-making with advanced machine learning techniques, it Saves time and improves efficiency in forensic workflows, provides accurate, reliable assessments.\
                        It also Enhances decision-making with clear, actionable insights.\
                        This tool is an invaluable asset for forensic labs, ensuring quality control and minimizing errors in sample analysis")
            
            uploaded_file = st.file_uploader(" Upload the sample dataset here in CSV Format", type="csv")

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("### The Sample Data is uploaded Successfully")
                st.write(df.head())
                prepared_data = prepare_data(df)
                st.subheader("Click the below button to train the AI Model with the uploaded dataset")
                if st.button("Click here to Train Model", type ="primary"):
                  train_model(prepared_data)
                if st.button ("Click here to Calculate M:F Ratio for the uploaded dataset", type ="primary"):
                    df_with_ratio = calculate_mf_ratio(df)
                    st.write("### Dataset with M:F Ratio")
                    st.write(df_with_ratio.head())
                    download_button(df_with_ratio, "Dataset_with_MF_Ratio.csv", "Download Dataset with M:F Ratio")     
                st.subheader("🔍 Assess Sample")
                sample_id = st.text_input(" ##### Enter the Sample ID here")
                if sample_id:
                   assess_sample(prepared_data,sample_id)
                
                #st.subheader("Calculate M:F Ratio")
               
                
          
                  


if __name__ == "__main__":
    main()
