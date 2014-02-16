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
