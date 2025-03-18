import pandas as pd

# Load the dataset
leftpunch_df = pd.read_csv("h_left_punch.txt")
rightpunch_df = pd.read_csv("h_right_punch.txt")

# Check for NaN values in the datasets
print("NaN values in left punch data:")
print(leftpunch_df.isna().sum())  # Count NaN values per column

print("\nNaN values in right punch data:")
print(rightpunch_df.isna().sum())  # Count NaN values per column

# Show rows with NaN values in the left punch dataset
print("Rows with NaN in left punch data:")
print(leftpunch_df[leftpunch_df.isna().any(axis=1)])

# Show rows with NaN values in the right punch dataset
print("\nRows with NaN in right punch data:")
print(rightpunch_df[rightpunch_df.isna().any(axis=1)])