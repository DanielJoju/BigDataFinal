import pyspark
from pyspark.context import SparkContext
from pyspark import SparkConf

conf = SparkConf()
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

# Load the adjacency list file
AdjList1 = sc.textFile("02AdjacencyList.txt")
print(AdjList1.collect())

AdjList2 = AdjList1.map(lambda line : line.split())
AdjList3 = AdjList2.map(lambda x : (int(x[0], [ int(y) for y in x[1:] ]))
AdjList3.persist()
print(AdjList3.collect())

nNumOfNodes = AdjList3.count()
print("Total Number of nodes")
print(nNumOfNodes)

# Initialize each page's rank; since we use mapValues, the resulting RDD will have the same partitioner as links
print("Initialization")
PageRankValues = AdjList3.mapValues(lambda node : 1 / float(nNumOfNodes))
print(PageRankValues.collect())

# Run 30 iterations
print("Run 30 Iterations")
for i in range(1, 30):
    print("Number of Iterations")
    print(i)
    JoinRDD = AdjList3.join(PageRankValues)
    print("join results")
    print(JoinRDD.collect())
    contributions = JoinRDD.flatMap(lambda (node, (adjList, ranking)) : [ (dest, ranking/len(adjList)) for dest in adjList])
    print("contributions")
    print(contributions.collect())
    accumulations = contributions.reduceByKey(lambda x, y : x + y)
    print("accumulations")
    print(accumulations.collect())
    PageRankValues = accumulations.mapValues(lambda v : 0.85 * v + 0.15 / float(nNumOfNodes))
    print("PageRankValues")
    print(PageRankValues.collect())

print("=== Final PageRankValues ===")
print(PageRankValues.collect())

# Write out the final ranks
PageRankValues.coalesce(1).saveAsTextFile("../Assignment2/PageRankValues_Final")

