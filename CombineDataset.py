import csv

# Replace these with the actual file paths to your CSV files
file_paths = ['BIDMC/combined_ecg_data', 'MITBIH/combined_ecg_data', 'ST-T/combined_ecg_data']

# Combine CSV files into one and add a label column which only have 0 and 1
with open('combined_ecg_data.csv', 'w', newline='\n') as combined_file:
    writer = csv.writer(combined_file)
    for F_name in file_paths:
        with open(f'{F_name}.csv', 'r', newline='\n') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[-1] =='0':
                    writer.writerow(row)
                elif row[-1] == '1': 
                    writer.writerow(row)
                elif int(row[-1]) > 1:
                    row[-1] = 1
                    writer.writerow(row)
                else:
                    print('Error' + row[-1])


# Combine CSV files into one directly and add label column which have its original label 
with open('combined_ecg_data2.csv', 'w', newline='\n') as combined_file:
    writer = csv.writer(combined_file)
    for F_name in file_paths:
        with open(f'{F_name}.csv', 'r', newline='\n') as f:
            reader = csv.reader(f)
            for row in reader:
                writer.writerow(row)
 