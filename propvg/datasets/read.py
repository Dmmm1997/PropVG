import json

path = 'data/seqtr_type/annotations/mixed-seg/instances_nogoogle_withid.json'

with open(path) as f:
    file = json.load(f)

print(1)