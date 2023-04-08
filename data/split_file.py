import json

# Set the input and output file paths
input_file = "vi_alpaca_reduced.jsonl"
output_prefix = "vi_alpaca_reduced.data/vi_alpaca_reduced"

# Set the number of lines per file
lines_per_file = 5000

# Open the input file and read its lines
with open(input_file, "r") as f:
    lines = f.readlines()

# Calculate the number of output files needed
num_files = len(lines) / lines_per_file

# Split the lines into chunks of lines_per_file
chunks = [lines[i:i+lines_per_file]
          for i in range(0, len(lines), lines_per_file)]

# Write each chunk to a separate output file
for i, chunk in enumerate(chunks):
    output_file = f"{output_prefix}_{i+1}.jsonl"
    with open(output_file, "w") as f:
        for line in chunk:
            f.write(line)
