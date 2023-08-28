import pandas as pd

def LoadParquet(parquet_file_path, drop_id=True):

    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file_path)
    
    if drop_id:
        df.drop(columns=[str(list(df.columns)[0])], inplace=True)

    return df