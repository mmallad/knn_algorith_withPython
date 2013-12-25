import operator
from numpy import *

#Create Dataset

groups = array([[2.5,3.5], [1.5,1],[4.5,4.5],[5,4.5],[3.5,2.5],[4,5]])
labels = ['Bad','Bad','Good','Good','Bad','Good']

def findRatings(pointsToFind, groups, labels, k):
	#pointsToFind is which we need to find label that is Good or Bad
	#groups is dataset which is used for training
	#labels is data classification name that is Good or Bad
	#k is the number of nearest neighbours used for voting
	#Now we need to find the label for pointsToFind that is it Good or Bad
	
	#First Lets Do Distance Calculation
	groupsSize = groups.shape[0]
	matDiff = tile(pointsToFind, (groupsSize, 1)) - groups
	sqMatDiff = matDiff**2
	sqDistances = sqMatDiff.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistInd = distances.argsort()
	classCount = {}
	#Now lets vote with lowest k distances
	for i in range(k):
		voteLabel = labels[sortedDistInd[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
	#Now Sort the dictionary
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	#Finally Return The Label At which pointsToFind Lies
	return sortedClassCount[0][0]

if __name__ == "__main__":
	print findRatings([4.1,5.0], groups, labels, 3)
