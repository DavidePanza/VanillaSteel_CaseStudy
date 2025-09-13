import pandas as pd
import numpy as np


# Compute Intersection over Union (IoU) similarity for two ranges
def dimension_similarity(min1, max1, min2, max2, type='iou'):
    """ Compute similarity between two ranges using IoU or distance-based metric. """
    # Handle NaN cases
    nan1 = pd.isna(min1) or pd.isna(max1)
    nan2 = pd.isna(min2) or pd.isna(max2)
    
    if nan1 and nan2:
        return 1.0  
    elif nan1 or nan2:
        return 0.0 
    
    if min1 == max1 and min2 == max2:
        return 1.0 if min1 == min2 else 0.0

    if type == 'iou':
        intersection = max(0, min(max1, max2) - max(min1, min2))
        union = max(max1, max2) - min(min1, min2)
        return intersection / union if union > 0 else 0
    elif type == 'distance':
        center1 = (min1 + max1) / 2
        center2 = (min2 + max2) / 2
        distance = abs(center1 - center2)
        return np.round(np.exp(-distance / max(abs(center1), abs(center2), 1)), 2)

    return 0

def run_dimension_similarity(row1, row2, dims, type='iou'):
    """ Compute average dimension similarity across multiple dimensions. """
    ious = []
    for dim in dims:
        min_col = f"{dim}_min"
        max_col = f"{dim}_max"
        iou = dimension_similarity(
            row1[min_col], row1[max_col],
            row2[min_col], row2[max_col], type=type
        )
        ious.append(iou)
    
    return sum(ious) / len(ious) if ious else 1.0


# Compute similarity for categorical variables
def categorical_similarity(val1, val2):
    """ Compute similarity for categorical values, handling NaNs. """
    if pd.isna(val1) and pd.isna(val2):
        return 1
    elif pd.isna(val1) or pd.isna(val2):
        return 0
    else:
        return 1 if val1 == val2 else 0

def run_categorical_similarity(row1, row2, dims):
    """ Compute average categorical similarity across multiple dimensions. """
    cat_matches = []
    for dim in dims:
        sim = categorical_similarity(row1[dim], row2[dim])
        if sim is not None:  
            cat_matches.append(sim)
    
    return sum(cat_matches) / len(cat_matches) if cat_matches else 0.0


# Compute similarity for numerical properties using specified distance metric
def run_grade_similarity(df, row1, row2, grade_properties_vars, distance_type='cosine'):
    """ Compute similarity between two rows based on numerical properties. """
    vec1, vec2 = [], []
    for col in grade_properties_vars:
        val1, val2 = row1[col], row2[col]
        if not (pd.isna(val1) or pd.isna(val2)):
            # Normalize by column range
            col_min = df[col].min()
            col_range = df[col].max() - col_min
            if col_range > 0:
                vec1.append((val1 - col_min) / col_range)
                vec2.append((val2 - col_min) / col_range)
    
    if not vec1:
        return 0.0
    
    vec1, vec2 = np.array(vec1), np.array(vec2)

    if distance_type == 'cosine':
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    elif distance_type == 'euclidean':
        distance = np.linalg.norm(vec1 - vec2)
        return 1 - (distance / np.sqrt(len(vec1)))
    return 0.0


# Combine all similarity measures with weights and handle ablations
def run_similarity_analysis(distance_type, similarity_type, ablations, weights, df, dimensions_vars, categorical_vars, grade_properties_vars):
    """ Run similarity analysis with specified parameters. """
    dimensions_matrix = np.zeros((len(df), len(df)))
    categories_matrix = np.zeros((len(df), len(df)))
    properties_matrix = np.zeros((len(df), len(df)))

    for i in range(len(df)):
        if i % 20 == 0:
            print(f"Processing row {i}/{len(df)}")
        for j in range(len(df)):
            if i != j:
                if 'dimensions' not in ablations:
                    sim_iou = run_dimension_similarity(df.iloc[i], df.iloc[j], dims=dimensions_vars, type=similarity_type)
                    dimensions_matrix[i, j] = sim_iou
                if 'categories' not in ablations:
                    sim_cat = run_categorical_similarity(df.iloc[i], df.iloc[j], dims=categorical_vars)
                    categories_matrix[i, j] = sim_cat
                if 'properties' not in ablations:
                    sim_prop = run_grade_similarity(df, df.iloc[i], df.iloc[j], grade_properties_vars=grade_properties_vars, distance_type=distance_type)
                    properties_matrix[i, j] = sim_prop
            else:
                if 'dimensions' not in ablations:
                    dimensions_matrix[i, j] = 1.0
                if 'categories' not in ablations:
                    categories_matrix[i, j] = 1.0
                if 'properties' not in ablations:
                    properties_matrix[i, j] = 1.0   

    combined_similarity = ((properties_matrix*weights[0] + categories_matrix*weights[1] + dimensions_matrix*weights[2])/(3-len(ablations))).round(3)

    return combined_similarity
