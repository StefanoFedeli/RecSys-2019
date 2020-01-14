outputs = []
with open("../../Outputs/TopPop_freeze.csv") as f:
    f.seek(18)
    for line in f:
        split = line.split(",")
        outputs.append(str(split[1]))
#print(outputs)
print(len(outputs))
occurrences = []
for item in outputs:
    occurrences.append(outputs.count(item))
print(occurrences)
occurrences.sort(reverse=True)
print(occurrences)

for i in range(0, 50):
    toRemove = occurrences[i]
    del occurrences[i+1:toRemove]
    print(occurrences)

print(occurrences)

