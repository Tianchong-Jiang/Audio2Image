from PIL import Image
import os
import csv
import pdb

str_to_filter = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

output = []

with open(".output/output/prompts_10per.csv", 'r') as f:
    reader = csv.reader(f)

    for row in reader:
        for str in str_to_filter:
            row[2] = row[2].replace(str, '')
        output.append(row)

# write to file
with open(".output/output/prompts_10per_filtered.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output)

print("Done")