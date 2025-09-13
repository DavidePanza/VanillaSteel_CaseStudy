# Setup and Usage Instructions

1. Clone the repository with notebooks and source code (src).
2. Run the notebooks in order: `task_1.ipynb` then `task_2.ipynb`
3. Output files will be generated in the `results/` directory

## Project Structure

```
├── notebooks/
│   ├── task_1.ipynb
│   └── task_2.ipynb
├── resources/
│   ├── task_1/
│   │   ├── supplier_data1.xlsx
│   │   ├── supplier_data2.xlsx
│   │   └── dataset_merged.csv
│   └── task_2/
│       ├── reference_properties.tsv
│       └── rfq.csv
├── src/
│   ├── data_processing.py
│   ├── data_visualisation.py
│   ├── generate_report.py
│   └── similarities_computation.py
└── results/
    ├── inventory_dataset.csv
    └── top3.csv
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```


**Notes**: A more detailed description is contained in the notebooks before each step. The following writing represents more an overview of the processing and analysis steps I took in the projects.

<br>
<br>

# TASK 1

### Data Loading 
- Loaded two supplier datasets (`supplier_data1.xlsx`, `supplier_data2.xlsx`) with identical row counts
- Identified inconsistent column naming and data type issues requiring standardization

### Label Formatting & Column Renaming
- Standardized all variable names to lowercase with underscores replacing spaces
- Corrected wrongly labelles variables based on domain research:
  - 'Quantity' --> 'a5/a80' (recognized as standard mechanical property of metals)
  - Generic 'Description' --> 'defect_description' and 'material_description'

### Data Type Standardization  
- Converted 'width_mm' to float types with 2 decimal precision to be consistent with 'thickness_mm' type
- Changed 'article_id' to object type (more appropriate for identifier data)
- Standardized text fields to lowercase for consistency
- Applied German translations to maintain language consistency:

### Data Quality Assessment
- Checked for missing values and duplicates across both datasets (none found)
- Performed visual inspection using pairplot analysis
- Identified zero values in mechanical properties ('rp02', 'rm', 'ag', 'a5/a80', 'ai')
- Decided to preserve zero values as they likely represent intrinsic material properties or non-applicable measurements

### Datasets Join
- Merged datasets using column-wise concatenation based on identical row counts (50 rows in both datasets)
- Acknowledged significant methodological concerns with this approach:
  - No unique reference key to guarantee row alignment
  - Unclear whether datasets represent same materials or different material categories
  - On the other hands, row-wise stacking would create sparse, artificial dataset (maybe an even worse choice)
- Decided to proceed with column-wise join while explicitly acknowledging its limitations

<br>
<br>



# TASK 2

## Data Processing Pipeline

### Data Loading & Cleaning
- Loaded RFQ data and reference properties, standardized column names (spaces to underscores, lowercase except chemical abbreviations)
- Fixed grade inconsistencies (HC380La → HC380LA, 26Mnb5 → 26MnB5) and merged datasets on material grades
- Verified complete grade matching (found 100% success rate) and removed empty/sparse columns
- Corrected data entry errors 

### Categorical Variables Processing  
- Imputed missing form values using dimensional patterns:
  - thickness_min = 22.0 → Coils
  - width = 1520 → Coils (based on observed patterns)
- Treated other missing categorical values as NaN for subsequent similarity analysis

### Grade Properties Engineering
- Parsed property strings to extract numerical ranges and bounds, creating midpoint variables
- Filtered variables by sparsity (<85% missing) to remove very spparse variables
- Imputed missing chemical composition values with 0, reasoning that missing properties likely indicate absence rather than unknown values
- I acknowledge this as a bold assumption but observation of consistent patterns across same steel grades support this approach

### Dimensions Processing
- Visualized min-max relationships to identify outliers and cleaning objectives
- Excluded very sparse dimensions (outer_diameter, yield_strength, tensile_strength)

**Diameter Corrections:**
- Moved misclassified outer_diameter = 610.0 to inner_diameter (research confirmed this error)
- Removed unrealistic outer_diameter values >2000 based on plausibility assessment
- Imputed missing inner_diameters with 610.0 for Coils/Strips/Slit Coils (standard industry dimension)

**Data Quality Fixes:**
- Corrected logical errors (thickness_min > thickness_max)
- Fixed unrealistic outliers: width = 8000, length >50000, weight >200000 by replacing or dividing by 10
- Suspected extra zeros in data entry for length/weight corrections
- Handled singleton dimensions by setting missing min/max equal when only one boundary existed

## Similarity Analysis Framework

#### Variable Selection
- **Grade Properties**: I select 12 properties (carbon, manganese, silicon, sulfur, phosphorus, vanadium, aluminum, titanium, niobium, tensile strength, yield strength, elongation)
  - I exclude variables with >85% sparsity
- **Categorical Variables**: I include 5 surface and form characteristics (coating, finish, surface_type, surface_protection, form)  
- **Dimensions**: I use 5 physical measurements (length, width, thickness, weight, inner_diameter)
  - I exclude variables with very high sparsity  

### Analysis Parameters & Methods

#### Grade Properties Similarity
- Applied euclidean distance for normalized property vectors (tested cosine similarity with similar results)
- Normalized each property column to [0,1] range using min-max scaling to account for different ranges across variables
- Skipped columns with missing values for each pairwise comparison
- Treated both-zero vectors as similar (1.0), one-zero vectors as dissimilar (0.0)

#### Categorical Similarity
- Used vectorized broadcasting for pairwise categorical comparisons
- Treated NaN-NaN as similar (1.0), NaN-value as dissimilar (0.0)
- Reasoned that missing categorical data represented meaningful similarity rather than dissimilarity
- Averaged matches across all categorical variables

#### Dimensional Similarity
- Used IoU (Intersection over Union) for dimensional range overlaps as explicitly requested
- Noted that most intervals were singletons or non-overlapping (typically 0 or 1 values)
- Acknowledged distance metrics would probably perform better for this data structure
- Averaged similarities across all dimensions

### Implementation & Output
- Set equal weights [1,1,1] across component types assuming equal importance
- Extracted top 3 most similar materials for each entry for procurement recommendations
- Exported results to CSV with RFQ ID pairs and similarity scores

## Exploratory Analysis

### Exploration Categories
- **Similarity Metrics**: Baseline euclidean, IoU dimensional overlap, cosine similarity
- **Ablation Studies**: Systematic removal of dimensions/categories/properties to assess contributions
- **Weighting Schemes**: Property-focused [3,1,1], category-focused [1,3,1], dimension-focused [1,1,3], clustering-optimized [6,3,1]

### Configuration Testing
- Sorted data by grade to reveal clustering patterns in visualizations
- Tested similarity metrics (euclidean vs cosine vs IoU), ablation studies, and weighting schemes

### Key Findings
- **Similarity Comparison**: Cosine similarity with distance metrics produced more evident clustering due to stronger discriminative effects
- **Component Importance**: Grade properties proved most effective for clustering steel categories
- **Reasoning**: Grade properties represented inherent material identity while dimensions and surface treatments were application-specific variables


## Observations

The main issue I encountered with the dataset was the high number of missing values. I tried to impute some of them, but most of the time I lacked a solid criterion to do so, although I was able to impute some of the missing `form` values. I also considered that for some grade and category variables, missing values are probably inherent to the properties and processing of the steel.  

By visualizing the data and consulting industry research, I was able to identify evident outliers and incorrect entries, and I put some effort into cleaning these using industry informations I gathered online. With more time I could have done a better job, as I most likely overlooked potential missing values. An approach to impute missing values I could have tried could have been to group entries along specific dimensions and use group-level statistics (e.g. mode) to infer the missing values.  

Regarding the choice of analysis, I mostly followed the indications in the task report. I found similarity calculations involving grades particularly challenging, as I was unsure how to deal with singleton values (often due to the presence of bounds). The lack of overlap in was also problematic, which occured in most of the comparisons. To partially address this, I propose using a distance-based measure.  

Another challenge concerned treatment of missing values in each similarity analysis. I mostly decided to consider these as informative missing values, treating the presence of two missing values as a match. This choice may not be universally correct, but in some cases it was supported by consistent patterns observed in the data.  

I spent time improving the functions to run the analysis efficiently. My first draft included brute-force computations, which were very inefficient. By using arrays and broadcasting, I significantly improved performance.  

Although data cleaning and preprocessing could be improved further, especially with deeper domain knowledge, I was able to visually identify clusters in the dataset. Sorting the data based on grade revealed that steels with related grades also share similar properties, and to a lesser extent, categories.  

Dimensions were the component with the least influence on clustering, as they relate more to specific applications rather than intrinsic material characteristics.
