import json

with open('data.json', 'r') as f:
    data = json.load(f)

rows = []
for row in data:
	values = str(row.values())
	values = values.split('{', 1)[1]
	values = '{' + values.split('\'')[0]

	rows.append(values)
        
with open('data_fixed.json', 'w') as f:
    # write the dictionary to JSON
    json.dump(rows, f)
    
f = open('data_fixed.json')

data = json.load(f)

print(data[0])