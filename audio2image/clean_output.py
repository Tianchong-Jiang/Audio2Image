import csv

output = []

with open(".output/output/raw_output_5.csv", 'r') as f:
    reader = csv.reader(f)

    last_name = ''
    last_epoch = ''

    for row in reader:
        if row[0] != last_name or row[1] != last_epoch:
            output.append(row)
        last_name = row[0]
        last_epoch = row[1]

# write to file
with open(".output/output/output_5.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(output)

print("Done")