import pandas as pd
import numpy as np


def get_top_n_similarities(similarity_matrix, n=5):
    """ Get top N similar items for each item from the similarity matrix. """
    similarity_no_perfect = similarity_matrix.copy()
    similarity_no_perfect[similarity_no_perfect == 1.0] = -np.inf
    top_n_indices = np.argsort(-similarity_no_perfect, axis=1)[:, :n]
    top_n_values = np.take_along_axis(similarity_matrix, top_n_indices, axis=1)
    return top_n_indices, top_n_values


def create_output_csv(df, top_n_indices, top_n_values, output_path):
    """ Create output CSV with similarity results. """
    results = []
    for idx in range(len(df)):
        similar_indices = top_n_indices[idx]
        similar_values = top_n_values[idx]
        for sim_idx, sim_val in zip(similar_indices, similar_values):
            results.append({
                'index':idx,
                'RFQ_ID': df.iloc[idx]['id'],
                'Similar_RFQ_ID': df.iloc[sim_idx]['id'],
                'Similarity_Score': round(float(sim_val), 3),
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"Similarity results saved to {output_path}")
