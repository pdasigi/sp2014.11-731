import gzip, pickle

fcounts = {}
ecounts = {}
for line in gzip.open('data/train.ru-en.align.gz'):
	parts = line.decode('utf-8').strip().split(" ||| ")
	fwords = parts[0].split()
	ewords = parts[1].split()
	aligns = [(int(x.split('-')[0]), int(x.split('-')[1])) for x in parts[2].split()]
	for i, j in aligns:
		if fwords[i] in fcounts:
			if ewords[j] in fcounts[fwords[i]]:
				fcounts[fwords[i]][ewords[j]] += 1
			else:
				fcounts[fwords[i]][ewords[j]] = 1
		else:
			fcounts[fwords[i]] = {ewords[j] : 1}
			
		if ewords[j] in ecounts:
			if fwords[i] in ecounts[ewords[j]]:
				ecounts[ewords[j]][fwords[i]] += 1
			else:
				ecounts[ewords[j]][fwords[i]] = 1
		else:
			ecounts[ewords[j]] = {fwords[i] : 1}

fprobs = {}
eprobs = {}
for fw in fcounts:
	s = sum(fcounts[fw].values())
	fprobs[fw] = {k: fcounts[fw][k]/float(s) for k in fcounts[fw]}
for ew in ecounts:
	s = sum(ecounts[ew].values())
	eprobs[ew] = {k: ecounts[ew][k]/float(s) for k in ecounts[ew]}

pout = open("lexprobs.pkl", "wb")
pickle.dump((fprobs, eprobs), pout)
pout.close()
