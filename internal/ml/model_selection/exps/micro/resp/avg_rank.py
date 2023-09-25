from pprint import pprint
import numpy as np

data = {
    "TFMEM":
        ["GradNorm", "NASWOT", "NTKCond", "NTKTrace", "NTKTrAppx", "Fisher", "GraSP", "SNIP", "ExpressFlow", "SynFlow"],

    "NB101+C10": [-0.34, 0.37, -0.28, -0.42, -0.53, -0.37, 0.14, -0.27, 0.38, 0.39],
    "NB201+C10": [0.64, 0.79, -0.48, 0.37, 0.34, 0.38, 0.53, 0.64, 0.78, 0.78],
    "NB201+C100": [0.64, 0.80, -0.39, 0.38, 0.38, 0.38, 0.54, 0.63, 0.77, 0.76, ],
    "NB201+IN-16": [0.57, 0.78, -0.41, 0.31, 0.36, 0.32, 0.52, 0.57, 0.75, 0.75],

    "DNN+Frappe": [0.45, 0.61, -0.77, 0.54, 0.13, 0.48, -0.27, 0.68, 0.79, 0.77],
    "DNN+UCI": [0.39, 0.63, -0.56, 0.37, 0.31, 0.21, -0.23, 0.62, 0.73, 0.68],
    "DNN+Criteo": [0.32, 0.69, -0.66, 0.46, 0.01, 0.41, -0.18, 0.78, 0.90, 0.74],

}


def average_rank(data, columns):
    # Extract data values based on specified columns and convert to numpy array
    values = np.array([data[col] for col in columns]).T

    # Take the absolute value of each element
    abs_data = np.abs(values)
    # Rank the data for each column such that largest value is ranked as 1
    ranks = (-abs_data).argsort(axis=0).argsort(axis=0) + 1

    # Calculate the average rank for each row
    average_ranks = ranks.mean(axis=1)
    return average_ranks


# Split the columns into the two categories
cv_columns = ["NB101+C10", "NB201+C10", "NB201+C100", "NB201+IN-16"]
tabular_columns = ["DNN+Frappe", "DNN+UCI", "DNN+Criteo"]

# Compute the average rank across the specified columns
avg_rank_cv = average_rank(data, cv_columns)
avg_rank_tabular = average_rank(data, tabular_columns)
avg_rank_all = average_rank(data, list(data.keys())[1:])  # excluding "TFMEM"

# Add these to the data dictionary
data["Avg Rank on CV"] = avg_rank_cv.tolist()
data["Avg Rank on Tabular"] = avg_rank_tabular.tolist()
data["Avg Rank on all dataset"] = avg_rank_all.tolist()

pprint(data)
