source = open('1-3.txt','r')
finetune = open('1-2.txt','r')
output = open('finetune.txt','w')
sl = [int(s) for s in source]
fl = [int(f) for f in finetune]
d = []
for s in sl:
    if s in fl or s-1 in fl or s+1 in fl:
        d.append(s)
for l in d:
    sl.remove(l)
for s in sl:
    print(s,file=output)
