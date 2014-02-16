import sys
import codecs
import operator
import pickle

def model1(train, initparam, numiter):
	param = initparam
	for s in range(numiter):
		transcounts = {}
		unicounts = {}
		for k in range(1, len(train)+1):
			(src, tgt) = train[k-1]
			srcwords = src.strip().split()
			tgtwords = tgt.strip().split()
			for i in range(1, len(srcwords)+1):
				sword = srcwords[i-1]
				fparam = param[sword]
				fparamsum = float(sum([fparam[word] for word in tgtwords]) + fparam['NULL'])
				for j in range(0, len(tgtwords)+1):
					if j == 0:
						tword = 'NULL'
					else:
						tword = tgtwords[j-1]
					delta = float(fparam[tword])/fparamsum
					if (tword, sword) in transcounts:
						transcounts[(tword, sword)] += delta
					else:
						transcounts[(tword, sword)] = delta
					if tword in unicounts:
						unicounts[tword] += delta
					else:
						unicounts[tword] = delta
		for (tword, sword) in transcounts:
			param[sword][tword] = transcounts[tword, sword] / unicounts[tword]
	return param

def model2(train, inittparam, initqparam, numiter):
	tparam = inittparam
	qparam = initqparam
	for s in range(numiter):
		transcounts = {}
		unicounts = {}
		qtranscounts = {}
		qunicounts = {}
		for k in range(1, len(train)+1):
			(src, tgt) = train[k-1]
			srcwords = src.strip().split()
			tgtwords = tgt.strip().split()
			m = len(srcwords)
			l = len(tgtwords)
			for i in range(1, len(srcwords)+1):
				sword = srcwords[i-1]
				ftparam = tparam[sword]
				paramsum = 0
				for j in range(0, len(tgtwords)+1):
					if j == 0:
						tword = 'NULL'
					else:
						tword = tgtwords[j-1]
					if (j, i, l, m) not in qparam:
						qparam[(j, i, l, m)] = 1.0/(l+1)
					paramsum += ftparam[tword]*qparam[(j, i, l, m)]
				for j in range(0, len(tgtwords)+1):
					if j == 0:
						tword = 'NULL'
					else:
						tword = tgtwords[j-1]
					delta = float(ftparam[tword]*qparam[(j, i, l, m)])/paramsum
					if (tword, sword) in transcounts:
						transcounts[(tword, sword)] += delta
					else:
						transcounts[(tword, sword)] = delta
					if tword in unicounts:
						unicounts[tword] += delta
					else:
						unicounts[tword] = delta
					if (j, i, l, m) in qtranscounts:
						qtranscounts[(j, i, l, m)] += delta
					else:
						qtranscounts[(j, i, l, m)] = delta
					if (i, l, m) in qunicounts:
						qunicounts[(i, l, m)] += delta
					else:
						qunicounts[(i, l, m)] = delta
		for (tword, sword) in transcounts:
			tparam[sword][tword] = transcounts[tword, sword] / unicounts[tword]
		for (j, i, l, m) in qtranscounts:
			qparam[(j, i, l, m)] = qtranscounts[(j, i, l, m)] / qunicounts[(i, l, m)]
	return (tparam, qparam)

def model1align(src, tgt, param):
	srcwords = src.split()
	tgtwords = tgt.split()
	alignments = {}
	for i in range(1, len(srcwords) + 1):
		srcword = srcwords[i-1]
		if srcword in param:
			aprobs = {}
			for x in tgtwords+['NULL']:
				if x in param[srcword]:
					aprobs[x] = param[srcword][x]
				else:
					aprobs[x] = 0
		else:
			aprobs = {x: 0 for x in tgtwords+['NULL']}
		besttgt = sorted(aprobs.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
		if besttgt != 'NULL':
			alignments[i] = tgtwords.index(besttgt)+1
		#print >>sys.stderr, srcword, '-->', besttgt, aprobs
	return alignments
	
def model2align(src, tgt, tparam, qparam):
	srcwords = src.split()
	tgtwords = tgt.split()
	m = len(srcwords)
	l = len(tgtwords)
	alignments = {}
	for i in range(1, len(srcwords) + 1):
		srcword = srcwords[i-1]
		if srcword in tparam:
			aprobs = {}
			for x in tgtwords+['NULL']:
				if x in tparam[srcword]:
					aprobs[x] = tparam[srcword][x]
				else:
					aprobs[x] = 0
		else:
			aprobs = {x : 0 for x in tgtwords+['NULL']}
		probs = {}
		if (0, i, l, m) in qparam:
			probs[('NULL', 0)] = aprobs['NULL']*qparam[(0, i, l, m)]
		else:
			probs[('NULL', 0)] = aprobs['NULL']*1.0/(l+1)
		j = 0
		for tgtword in tgtwords:
			j += 1
			if (j, i, l, m) in qparam:
				probs[(tgtword, j)] = aprobs[tgtword]*qparam[(j, i, l, m)]
			else:
				probs[(tgtword, j)] = aprobs[tgtword]*1.0/(l+1)
		(besttgt, j) = sorted(probs.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
		if besttgt != 'NULL':
			alignments[i] = j
		#print >>sys.stderr, srcword, '-->', besttgt, aprobs
	return alignments
	
	
def heuristicinit(train):
	numswitht = {}
	twiths = {}
	swords = []
	#twords = []
	param = {}
	for (src, tgt) in train:
		srcwords = list(set(src.strip().split()))
		swords += srcwords
		tgtwords = list(set(tgt.strip().split()))
		#twords += tgtwords
		for word in tgtwords:
			if word in numswitht:
				numswitht[word]+=len(srcwords)
			else:
				numswitht[word] = len(srcwords)
		for word in srcwords:
			if word in twiths:
				twiths[word] = twiths[word].union(set(tgtwords))
			else:
				twiths[word] = set(tgtwords)
	swords = list(set(swords))
	#twords = list(set(twords))
	nullalignprob = 1.0 / float(len(swords))
	for word in swords:
		unitgtdist = {}
		for tgtword in twiths[word]:
			unitgtdist[tgtword] = 1.0 / float(numswitht[tgtword])
		unitgtdist['NULL'] = nullalignprob
		param[word] = unitgtdist
	return param	

def qinit(maxl, maxm):
	qparam = {}
	for m in range(maxm):
		for l in range(maxl):
			for i in range(1, m+1):
				for j in range(0, l+1):
					qparam[(j, i, l, m)] = 1.0 / (l+1)
	return qparam

"""
def helper():
	print >>sys.stderr,  "Usage:  python ibm-models-pe.py [mode] [modelname] [modelfile] [source] [target] [init-model/alignment]"
	print  >>sys.stderr,  "\tmode = train/train-rev/align/align-rev"
	print  >>sys.stderr,  "\tmodelname = model1/model2 (more to come!)"
	print  >>sys.stderr,  "\tmodelfile = name of file to write (for train) or read (for align)"
	print  >>sys.stderr,  "\tinfile = trainfile (for train) or testfile (for align)"
	print  >>sys.stderr,  "\tinit-model/alignment = Initial model file (for model2 train) or alignment output (for align)"

if len(sys.argv) < 4:
	print >>sys.stderr, "Error: Not enough arguments!"
	helper()
	sys.exit(1)
mode = sys.argv[1]
if (mode not in ["train", "train-rev", "align", "align-rev"]) or (mode in ["align", "align-rev"] and len(sys.argv) < 6):
	print >>sys.stderr, "Error: Incorrect mode or arguments!"
	helper()
	sys.exit(1)
model = sys.argv[2]
if (model != "model1" and model != "model2"):
	print >>sys.stderr, "Error: Only model1 and model2 are supported!"
	helper()
	sys.exit(1)
if (model == "model2" and mode == "train" and len(sys.argv) < 5):
	print >>sys.stderr, "Error: Training model2 requires input model2 file"
	helper()
	sys.exit(1)
modelfile = sys.argv[3]

if mode == "train" or mode == "train-rev":
	trainfile = codecs.open(sys.argv[4], 'r', 'utf-8')
	maxm = 0
	maxl = 0
	train = []
	for line in trainfile:
		parts = line.strip().split(" ||| ")
		if mode == "train":
			f = parts[0]
			e = parts[1]
		else:
			f = parts[1]
			e = parts[0]

		train.append((parts[0], parts[1]))
		l = len(f.split(" "))
		m = len(e.split(" "))
		if m > maxm:
			maxm = m
		if l > maxl:
			maxl = l
		train.append((f, e))

	#print >>sys.stderr, "Loaded training data"
	numiter = 8
	if model == "model1":
		initparam = heuristicinit(train)
		model1param = model1(train, initparam, numiter)
		model1file = open(modelfile, "wb")
		pickle.dump(model1param, model1file)
	elif model == "model2":
		initqparam = qinit(maxl, maxm)
		print >>sys.stderr, "Initiated distortion parameters"
		if len(sys.argv) > 5:
			model1file = open(sys.argv[5], "rb")
			model1param = pickle.load(model1file)
			print >>sys.stderr, "Loaded model1 to memory"
			model2param = model2(train, model1param, initqparam, numiter)
		else:
			initparam = heuristicinit(train)
			model2param = model2(train, initparam, initqparam, numiter)
		model2file = open(modelfile, "wb")
		pickle.dump(model2param, model2file)
elif mode == "align" or mode == "align-rev":

	if model == 'model1':
		model1file = open(modelfile, "rb")
		model1param = pickle.load(model1file)
		print >>sys.stderr, "Loaded model1 to memory"
	else:
		model2file = open(modelfile, "rb")
		(tparam, qparam) = pickle.load(model2file)
		print >>sys.stderr, "Loaded model2 to memory"

	testfile = codecs.open(sys.argv[4], 'r', 'utf-8')
	if mode == "align":
		test = [(line.split(" ||| ")[0].lower(), line.split(" ||| ")[1].strip().lower()) for line in testfile]
	else:
		test = [(line.split(" ||| ")[0].lower(), line.split(" ||| ")[1].strip().lower()) for line in testfile]
	out = open(sys.argv[5], 'w')
	k = 0
	for (src, tgt) in test:
		k += 1
		if model == 'model1':
			alignments = model1align(src.strip(), tgt.strip(), model1param)
		else:
			alignments = model2align(src.strip(), tgt.strip(), tparam, qparam)
		algn = []
		for i in alignments:
			algn.append("%d-%d"%(i-1, alignments[i]-1))
		print >>out, " ".join(algn)
"""
