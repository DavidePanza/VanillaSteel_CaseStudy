import pandas as pd
import numpy as np
import re

def extract_ranges(df, col_name):
    """ Extract midpoints from range strings in a column and create a new column with these midpoints. """
    mid_col_name = f"{col_name}_mid"  
    df[mid_col_name] = np.nan  
    
    for row in range(len(df)):
        value = df.iloc[row][col_name]
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        value_str_clean = re.sub(r"[^\d.\-<≥>≤]", "", value_str)
        
        if '-' in value_str_clean:
            parts = value_str_clean.split('-')
            try:
                low, high = float(parts[0]), float(parts[1])
                df.loc[df.index[row], mid_col_name] = (low + high) / 2 
            except ValueError:
                pass
        
        elif '<' in value_str_clean or '≤' in value_str_clean or '>' in value_str_clean or '≥' in value_str_clean:
            try:
                num_value = float(value_str_clean.replace('<', '').replace('≤', '').replace('>', '').replace('≥', '').strip())
                df.loc[df.index[row], mid_col_name] = num_value  
            except ValueError:
                pass
        else:
            df.loc[df.index[row], mid_col_name] = np.nan  
            print(f"Unexpected value format: {value} found in variable {col_name} at row {row}")

    return df


def fill_singleton_dimensions(df, col):
    """ Fill missing min/max in a column by assuming singleton ranges. """
    min_col = f"{col}_min"
    max_col = f"{col}_max"

    # If min is missing and max is present, set min to max
    df.loc[
        df[min_col].isna() & df[max_col].notna(),
        min_col] = df[max_col]

    # If max is missing and min is present, set max to min
    df.loc[
        df[max_col].isna() & df[min_col].notna(),
        max_col] = df[min_col]
    
    return df
   