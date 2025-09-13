import pandas as pd
import numpy as np


# Compute similarity matrix for grade properties
def grade_similarity(df, grade_properties_vars, distance_type='cosine'):
    """ Compute similarity matrix for grade properties using specified distance metric. """
    n = len(df)
    similarity_matrix = np.zeros((n, n))
    
    # normalise data
    data_normalized = df[grade_properties_vars].copy()
    for col in grade_properties_vars:
        col_min = data_normalized[col].min()
        col_range = data_normalized[col].max() - col_min
        if col_range > 0:
            data_normalized[col] = (data_normalized[col] - col_min) / col_range
        else:
            data_normalized[col] = 0
    
    # convert to numpy arrays 
    data_array = data_normalized.to_numpy()
    is_nan = pd.isna(data_normalized).values
    
    # compute similarities
    for i in range(n):
        if i % 250 == 0:
            print(f"Processing row {i}/{n}")
        similarity_matrix[i, i] = 1.0  # add diagonal == 1

        for j in range(i + 1, n):
            # find columns valid for both rows
            valid_mask = ~(is_nan[i] | is_nan[j])
            
            if not valid_mask.any():
                similarity = 0.0
            else:
                vec1 = data_array[i, valid_mask]
                vec2 = data_array[j, valid_mask]
                
                if distance_type == 'cosine':
                    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                    if norm1 == 0 and norm2 == 0:
                        similarity = 1.0   # both vectors zero --> similarity 1
                    elif norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                elif distance_type == 'euclidean':
                    distance = np.linalg.norm(vec1 - vec2)
                    similarity = 1 - (distance / np.sqrt(len(vec1)))
                else:
                    similarity = 0.0
            
            # fill matrix symmetrically
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix


# Compute similarity matrix for categorical variables
def categorical_similarity(df, categorical_vars):
    """Vectorized computation of categorical similarities."""
    n = len(df)
    similarity_matrix = np.zeros((n, n))
    
    for var in categorical_vars:
        values = df[var].to_numpy()
        # NaN values mas
        nan_mask = pd.isna(values)
        
        # broadcasting to create comparison matrix
        matches = (values[:, None] == values[None, :])
        
        # Set NaN == NaN --> 1, NaN != value --> 0
        nan_both = nan_mask[:, None] & nan_mask[None, :]
        nan_either = (nan_mask[:, None] | nan_mask[None, :]) & ~nan_both
        
        # Get final matches
        matches = matches & ~nan_either  
        matches = matches | nan_both     
        similarity_matrix += matches.astype(float)
    
    # average across variables
    return similarity_matrix / len(categorical_vars)


# Compute similarity matrix for dimension ranges
def dimension_similarity(df, dimensions_vars, type='iou'):
    """ Compute similarity matrix for dimensional ranges. """
    n = len(df)
    total_similarity = np.zeros((n, n))
    
    for dim in dimensions_vars:
        mins = df[f"{dim}_min"].to_numpy()
        maxs = df[f"{dim}_max"].to_numpy()
        nan_mask = pd.isna(mins) | pd.isna(maxs)
        singletons_mask = (mins == maxs) & ~nan_mask
        
        # create matrix for this dimension
        dim_similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n): 
                if i == j:
                    sim = 1.0
                elif nan_mask[i] and nan_mask[j]:
                    sim = 1.0
                elif nan_mask[i] or nan_mask[j]:
                    sim = 0.0
                elif singletons_mask[i] and singletons_mask[j]:
                    sim = 1.0 if mins[i] == mins[j] else 0.0
                else:
                    if type == 'iou':
                        intersection = max(0, min(maxs[i], maxs[j]) - max(mins[i], mins[j]))
                        union = max(maxs[i], maxs[j]) - min(mins[i], mins[j])
                        sim = intersection / union if union > 0 else 0
                    elif type == 'distance':
                        center_i = (mins[i] + maxs[i]) / 2
                        center_j = (mins[j] + maxs[j]) / 2
                        distance = abs(center_i - center_j)
                        sim = np.round(np.exp(-distance / max(abs(center_i), abs(center_j), 1)), 2)
                    else:
                        sim = 0.0
                
                dim_similarity[i, j] = sim # this is because of symmetry
                dim_similarity[j, i] = sim

        # add this dimension to total
        total_similarity += dim_similarity
    
    return total_similarity / len(dimensions_vars)


# Main function to compute combined similarity matrix
def run_similarity_analysis(df, dimensions_vars, categorical_vars, grade_properties_vars, 
                           distance_type='cosine', similarity_type='iou', 
                           ablations=[], weights=[1, 1, 1]):
    """ Run similarity analysis combining different components with optional ablations. """
    n = len(df)
    matrices = []
    used_weights = []
    
    if 'properties' not in ablations and grade_properties_vars:
        print("Computing property similarities")
        properties_matrix = grade_similarity(df, grade_properties_vars, distance_type)
        matrices.append(properties_matrix)
        used_weights.append(weights[0])
    
    if 'categories' not in ablations and categorical_vars:
        print("Computing categorical similarities")
        categories_matrix = categorical_similarity(df, categorical_vars)
        matrices.append(categories_matrix)
        used_weights.append(weights[1])
    
    if 'dimensions' not in ablations and dimensions_vars:
        print("Computing dimensional similarities")
        dimensions_matrix = dimension_similarity(df, dimensions_vars, type=similarity_type)
        matrices.append(dimensions_matrix)
        used_weights.append(weights[2])
    
    # Weighted combination
    combined_similarity = np.zeros((n, n))
    total_weight = sum(used_weights)
    
    for matrix, weight in zip(matrices, used_weights):
        combined_similarity += matrix * (weight / total_weight)
    
    return np.round(combined_similarity, 3)


def run_explorations(explorations, default_params):
    """ Run multiple similarity analyses with different parameters. """
    matrices = {}
    for exp_name, overrides in explorations.items():
        print(f"Running analysis for: {exp_name}")
        params = {**default_params, **overrides}
        matrices[exp_name] = run_similarity_analysis(
            params['df'], params['dimensions_vars'], params['categorical_vars'], 
            params['grade_properties_vars'], distance_type=params['distance_type'],
            similarity_type=params['similarity_type'], ablations=params['ablations'],
            weights=params['weights']
        )
    return matrices