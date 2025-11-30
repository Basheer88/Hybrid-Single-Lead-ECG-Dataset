import matplotlib.pyplot as plt
import pandas as pd

# Read your CSV file
data = pd.read_csv("combined_ecg_data.csv")

# Assuming the last column stores numerical values
values = data.iloc[:, -1].tolist()

# Get unique values and their counts
value_counts = pd.Series(values).value_counts()

# Calculate percentages
total_count = len(data)
percentages = (value_counts / total_count) * 100

# Prepare labels and values
labels = value_counts.index.astype(str).tolist()  # Convert value index to strings
values = percentages.tolist()

# Create the pie chart
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart of Value Percentages (Last Column)')
plt.show()