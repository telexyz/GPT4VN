import json
t = json.load(open('data.json'))
for d in t:
	for vi in d['translate']:
		print(vi[4:])
	print()
