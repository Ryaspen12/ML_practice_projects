import pandas as pd
import os

def load_data(file_path):
    """
    Load the subject data from a dat file.

    Parameters:
    file_path (str): The path to the dat file containing the subject data.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded gait data.
    """
    try:
        data = pd.read_csv(file_path, sep=r"\s+", header=None)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def label_dataframe(raw_df):
    """
    Label the DataFrame columns based on the PAMAP2 dataset specifications.

    Parameters:
    raw_df (pd.DataFrame): The raw DataFrame to be labeled.

    Returns:
    pd.DataFrame: A DataFrame with labeled columns.
    """

    base_cols = ["timestamp", "activity", "heart_rate"]
    imu_cols = [
    "temp",
    "acc1_x","acc1_y","acc1_z",
    "acc2_x","acc2_y","acc2_z",
    "gyro_x","gyro_y","gyro_z",
    "mag_x","mag_y","mag_z",
    "ori1","ori2","ori3","ori4"
    ]

    column_names = (base_cols +
    ["hand_" + c for c in imu_cols] +
    ["chest_" + c for c in imu_cols] +
    ["ankle_" + c for c in imu_cols]
    )

    if len(raw_df.columns) != len(column_names):
        print("Warning: The number of columns in the DataFrame does not match the expected number of column names.")
    
    raw_df.columns = column_names[:len(raw_df.columns)]
    print("DataFrame columns labeled successfully.")
    df = raw_df.copy()
    return df


def clean_data(df):
    """
    Process the DataFrame by handling missing values, dropping activity = 0 labels, 
    dropping orientation columns, and performing any other necessary cleaning steps.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.

    Returns:
    pd.DataFrame: A cleaned DataFrame ready for analysis.
    """
    # Handle missing values
    df = df.dropna()  # Drop rows with missing values (you can also choose to fill them instead)
    
    # Drop rows where activity = 0
    df = df[df['activity'] != 0]

    # Drop orientation columns if they exist
    orientation_cols = [col for col in df.columns if 'ori' in col]
    df = df.drop(columns=orientation_cols, errors='ignore')
    
    print("Data processed successfully.")
    return df

def load_all_subjects(data_dir):
    """Load and process data for all subjects in the specified directory based on .dat files."""
    all_subjects_data = []
    subject_num = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".dat"):
            file_path = os.path.join(data_dir, filename)
            raw_df = load_data(file_path)
            if raw_df is not None:
                labeled_df = label_dataframe(raw_df)
                cleaned_df = clean_data(labeled_df)
                cleaned_df['subject'] = subject_num  # Add a column to identify the subject
                all_subjects_data.append(cleaned_df)
                print(f"Subject {subject_num} data loaded and processed.")
                subject_num += 1
    return pd.concat(all_subjects_data, ignore_index=True)