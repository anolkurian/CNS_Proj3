import re

# Open the text file
with open('results.txt', 'r') as f:
    content = f.read()

# Parse the content and store the values in a list of tuples
results = []
prev_line= ''
for line in content.split('\n'):
    # print(line)
    if line.strip() != '':
        match = re.findall(r'([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)', line)
        if(len(match) > 0):
            match = match[0]
            # print(match[0])
            # print(match[2])
            results.append((float(match[0]), float(match[2]), prev_line))
        else:
            prev_line = line


diff_list = [(r[0] - r[1], r[2]) for r in results]
highest_diff = max(diff_list, key=lambda x: x[0])
lowest_diff = min(diff_list, key=lambda x: x[0])

print('Command with highest difference: ', highest_diff[1])
print('Command with lowest difference: ', lowest_diff[1])

# Command with highest difference:  Command: python3 hw.py problem1 simple,32,0.01 800
# Command with lowest difference:  Command: python3 hw.py problem1 simple,32,0.01 100