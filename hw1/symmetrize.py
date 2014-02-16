import sys

def intersect(f2e, e2f):
	#return f2e.intersection(set([(v, k) for (k, v) in e2f]))
	return f2e.intersection(e2f)

def union(f2e, e2f):
	#return f2e.union(set([(v, k) for (k, v) in e2f]))
	return f2e.union(e2f)

def growDiag(f2e, e2f):
	newf2e = list(intersect(f2e, e2f))
	unionAl = list(union(f2e, e2f))
	prevf2e = []
	while set(prevf2e) != set(newf2e):
		prevf2e = list(newf2e)
		for (i, j) in prevf2e:
			neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
			for (i1, j1) in neighbors:
				if (i1, j1) in unionAl:
					if (len(filter(lambda x: x[1] == j1, prevf2e))) == 0 or (len(filter(lambda x: x[0] == i1, prevf2e))) == 0:
						newf2e.append((i1, j1))
	return set(newf2e)

def growDiagFinal(f2e, e2f):
	unionAl = list(union(f2e, e2f))
	grown = list(growDiag(f2e, e2f))
	for (i, j) in unionAl:
		if (len(filter(lambda x: x[1] == j, grown))) == 0 or (len(filter(lambda x: x[0] == i, grown))) == 0:
			grown.append((i, j))
	return set(grown)
		

def growDiagFinalAnd(f2e, e2f):
	unionAl = list(union(f2e, e2f))
	grown = list(growDiag(f2e, e2f))
	for (i, j) in unionAl:
		if (len(filter(lambda x: x[1] == j, grown))) == 0 and (len(filter(lambda x: x[0] == i, grown))) == 0:
			grown.append((i, j))
	return set(grown)


def readAlignments(fileh):
	alignments = {}
	for ind, line in enumerate(fileh):
		aligns = line.strip().split(" ")
		sentaligns = set()
		for algn in aligns:
			if algn == "":
				continue
			s, t = algn.split("-")
			sentaligns.add((int(s), int(t)))
		alignments[ind] = sentaligns
	return alignments

"""
f2efile = open(sys.argv[1])
e2ffile = open(sys.argv[2])
outfile = open(sys.argv[3], "w")
f2ealignments = readAlignments(f2efile)
e2falignments = readAlignments(e2ffile)

for k in range(len(f2ealignments)):
	#print >>sys.stderr, "intersection", intersect(f2ealignments[k], e2falignments[k])
	#print >>sys.stderr, "union", union(f2ealignments[k], e2falignments[k])
	#print >>sys.stderr, "growDiag", growDiag(f2ealignments[k], e2falignments[k])
	#print >>sys.stderr, "growDiagFinal", growDiagFinal(f2ealignments[k], e2falignments[k])
	grownf2e = union(f2ealignments[k], e2falignments[k])
	#print f2ealignments[k], e2falignments[k], grownf2e
	grownaligns = []
	for (i, j) in grownf2e:
		grownaligns.append("%d-%d"%(i, j))
	print >>outfile, " ".join(grownaligns)
"""
