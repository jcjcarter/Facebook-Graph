import bookgraphs, comp182, hw3_testutil, provided, collections, numpy

def bfs(g, startnode):
    """
    Perform a breadth-first search on g starting at node startnode.

    Arguments:
    g -- undirected graph
    startnode - node in g to start the search from

    Returns:
    A tuple containing the distances from startnode to each
    node, the predecessors of each node in the BFS search
    tree, and the set of visited nodes, and a tuple containing the
    shorts number of paths.
    """
    dist = {}
    pred = {}
    numpath = {}

    # Initialize distances and predecessors
    for node in g:
        dist[node] = float('inf')
        pred[node] = None
        numpath[node] = 0
    dist[startnode] = 0
    numpath[startnode] = 1

    # Initialize visited set and search queue
    visited = set()
    queue = collections.deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        j = queue.popleft()
        for h in (g[j] - visited):
            if dist[h] == float('inf'):
                dist[h] = dist[j] + 1
                numpath[h] = numpath[j] #accounts for the first shortest path
                pred[h] = j
                queue.append(h)
            elif dist[h] == dist[j] + 1: #tests to see if there is another shortest path to node
                numpath[h] = numpath[h] + numpath[j]
        visited.add(j)
        
    return (dist, pred, visited, numpath)

def testrunbfs():
    """Tests to see if everything is correct with bfs function using the
    function testutils.

    Arguments:
    None

    Returns:
    True if the test passes and falses if the test fails."""
    a,d,c,data1 = bfs(testgraph1,6)
    data2 = {0: 1, 1: 3, 2: 1, 3: 2, 4: 1, 5: 1, 6: 1}
    print hw3_testutil.test_edge_to_fp_dict_compare(data1,data2,10e-6)
    
testgraph1 = {0:set([1,2]),
              1:set([0,3]),
              2:set([0,3,4]),
              3:set([1,2,5]),
              4:set([2,5,6]),
              5:set([3,4]),
              6:set([4])}
(a,b,c,d) = bfs(testgraph1, 6) #Used to test compute_flow and betweenness

def compute_flow(g, dist, npaths):
    """Computes the flow across all of the edges using the distance
    dictionary and number of shortest paths from BFS.

    Arguments:
    g -- undirected graph
    npaths -- a dictionary containing the number of shortest paths
    dist -- distance from start node to end

    Returns:
    A dictionary with edges as tuples as the keys and the flow as values."""

    edgeflows = {}
    furthestnodes = set()
    processededge = set()
    unprocessednodes = set()
    processednodes = set()
    value = 0
    for node in g:
        unprocessednodes.add(node)

    for i in g: #makes a complete graph with edges as keys
        for j in g:
            if j in g[i]:
                edgeflows[frozenset([i,j])] = 0

    maxdistance = max(dis for dis in dist.values() if dis != float('inf'))
    while maxdistance >= 0:
        
        for i in (edge for edge in g if dist[edge] == maxdistance): #finds all the furthest nodes first
            
            value = sum(edgeflows[frozenset([j,i])] for j in g[i] if dist[j] > dist[i])
                          
            calflow = (1+value)/float(npaths[i])
            #print "calflow", calflow
            for node in g[i]: #adds the computed flow value to each edge
                if dist[node] < dist[i]:
                    edgeflows[frozenset([node,i])] = (calflow*float(npaths[node]))
        maxdistance = maxdistance -1
    

    
    return edgeflows


def testruncomputeflow():
    """Tests to see if everything is correct with compute flow function using the
    function testutils.

    Arguments:
    None

    Returns:
    True if the test passes and falses if the test fails."""
    a,d,c,data1 = bfs(testgraph1,6)
    data2 = compute_flow(testgraph1,a,data1)
    data3 = {frozenset([2, 4]): 3.1666666666666665, frozenset([3, 5]): 0.8333333333333333, frozenset([2, 3]): 0.8333333333333333, frozenset([0, 2]): 1.3333333333333333, frozenset([1, 3]): 0.6666666666666666, frozenset([4, 6]): 6.0, frozenset([4, 5]): 1.8333333333333333, frozenset([0, 1]): 0.3333333333333333}
    print hw3_testutil.test_edge_to_fp_dict_compare(data2,data3,10e-6)
            
def test_compute_flow():
    """ Tests the functionality of the function compute flow
    using figure 3.20 and source node A.

    Arguments:
    None

    Returns:
    The result of the compute flow on figure 3.20 using source node A."""

    g = bookgraphs.fig3_18g
    (a,b,c,d) = bfs(g,'A')
    return compute_flow(g,a,d)
    

def find_max_distance(unprocessednodes, distdic):
    """ Finds the max distance value for the nodes remaining in the distance dictionary.

    Arguments:
    unprocessednodes -- a set of integers
    distdic -- a dictionary containing the distance from a source node to other nodes

    Returns:
    An integer that is the great distance of the remaining nodes"""

    maxdic = {}
    for node in unprocessednodes:
        maxdic[node] = distdic[node]
    maxdistance = max(maxdic.values())

    return maxdistance

def shortest_path_edge_betweenness(g):
    
    """Computes the shortest-path based betweenness of all edges of graph
    g by summing, for each edge, the scores that the edge receives
    from all runs of compute_flow.

    Arguments:
    g -- undirected graph
    

    Returns:
    A dictionary of all the edges of g with the scores that the edge
    receives from all the summation."""
    
    shortestpaths = {}
    for node in g: #sets up the dictionary for the edges as keys
        for nbr in g[node]:
            if nbr in g[node]:
                shortestpaths[frozenset([node,nbr])] = 0
        
    for node in g: #calls bfs for every node
        dist, pred, visited, npaths = bfs(g,node)
        flow = compute_flow(g,dist,npaths)
        for node in flow:
            
            shortestpaths[node] = shortestpaths[node] + flow[node]
        #print "Calculated Betweenness:", shortestpaths
    return shortestpaths

def testrunshortestpathedgebetweenness():
    """Tests to see if everything is correct with shortest path edge betweenness function using the
    function testutils.

    Arguments:
    None

    Returns:
    True if the test passes and falses if the test fails."""
    
    data2 = shortest_path_edge_betweenness(testgraph1)
    data3 = {frozenset([2, 4]): 14.333333333333332, frozenset([3, 5]): 9.666666666666666, frozenset([2, 3]): 9.0, frozenset([0, 2]): 10.666666666666666, frozenset([1, 3]): 9.333333333333332, frozenset([4, 6]): 12.0, frozenset([4, 5]): 9.0, frozenset([0, 1]): 5.999999999999999}
    print hw3_testutil.test_edge_to_fp_dict_compare(data2,data3,10e-6)


def betweenness_function_test():
    """Tests the functionality of the shorth path edge betweenness function with
    figure 3.20.

    Arguments:
    None

    Returns:
    Results from for the betweenness among the edges of figure 3.20."""

    return shortest_path_edge_betweenness(bookgraphs.fig3_18g)
    

def compute_q(g,c):
    """
    Computes the value of G, as given by equation (3), from the set c
    of communities of graph g. That is, c, is a list of sets of nodes
    that form a partition of the nodes of graph.

    Arguments:
    g -- undirected graph
    c - a list of sets of nodes that form a partion of the nodes of g.

    Returns:
    A single floating point number which is the value of Q. 
    """
    D = compute_d(g,c)
    square = numpy.dot(D,D) 
    Tr = 0
    for i in xrange(len(c)):
        Tr = Tr + float(D[i,i])
    Q = Tr - numpy.sum(square)
    return Q

def testruncomputeq():
    """Tests to see if everything is correct with the compute q function using the
    function testutils.

    Arguments:
    None

    Returns:
    True if the test passes and falses if the test fails."""
    c = connected_components(bookgraphs.fig3_15g)
    data2 = compute_q(bookgraphs.fig3_15g, c)
    data3 = 2.2204460492503131e-16
    print hw3_testutil.test_fp_compare(data2,data3,10e-6)

def compute_d(g,c):
    """ Helper function that computes the value of fraction of
    all edges in the graph that connect nodes in component i to nodes
     in component j.

     Arguments:
     g -- undirected graph in dictionary form
     c -- a list of sets of nodes that form a partion of the nodes of g

     Returns:
     A matrix for the fraction of all edges in the graph g."""
    
    edges = edge_count(g)
    D = numpy.zeros((len(c),len(c)))
    for i, a in enumerate(c):
        
        for elem in a:
            
            for j, b in enumerate(c):
                #compfun = node_index(c,nbr)
                #D[i][compfun] += 1
                D[i,j] += (1.0/float(edges))*len(g[elem] & b)
    #print D    
    for i in range(len(c)): #accounts for the double counting of edges within the matrix
        D[i,i] /= 2.0
    #print D
        
    return D
def edge_count(graph):
    """
    Returns the number of edges in a graph.

    Arguments:
    graph -- The given graph.

    Returns:
    The number of edges in the given graph.
    """
    edge_double_count = 0
    for nodeKey in graph.keys():
        edge_double_count = edge_double_count + len(graph[nodeKey])

    return edge_double_count / 2
def node_index(c,node):
    """Helper function that indexes a node.
    Arguments:
    c -- a list of sets of nodes that form a partition
    node -- a node to be found in a partition
    
    Returns:
    The inputed node if the node is found in the connected-components.
        """
    for i in xrange(len(c)):
        if node in c[i]:
            return i
    return "Node not found"

def connected_components(g):
    """
    Find all connected components in g.

    Arguments:
    g -- undirected graph

    Returns:
    A list of sets where each set is all the nodes in
    a connected component.
    """
    # Initially we have no components and all nodes remain to be
    # explored.
    components = []
    remaining = set(g.keys())

    while remaining:
        # Randomly select a remaining node and find all nodes
        # connected to that node
        node = list(remaining)[0]
        visited = bfs(g, node)[2]
        components.append(visited)

        # Remove all nodes in this component from the remaining nodes
        remaining -= visited

    return components

def test_compute_q():
    """Tests the functionality of compute q on figures 3.14 and 3.15.

    Arguments:
    None

    Returns:
    The Q for figures 3.14, 3.15, and g1. """

    a = bookgraphs.fig3_14g
    b = bookgraphs.fig3_15g
    c = testgraph1
    cc = connected_components(c)
    ac = connected_components(a)
    bc = connected_components(b)
    print "g1 Q Value:", compute_q(c,cc)
    print ""
    print "3.14 Q value:", compute_q(a,ac)
    print ""
    print "3.15 Q value:", compute_q(b,bc)

def run_gn_graph_partition_test():
    """Runs the gn_graph_parition function on figures
    3.14, 3.15, and g1.

    Arguments:
    None

    Returns:
    The results of the Q value and communities from the 3.14 and 3.15 figures and g1. """
    print gn_graph_partition(testgraph1)
    print ""
    print gn_graph_partition(bookgraphs.fig3_14g)
    print ""
    print gn_graph_partition(bookgraphs.fig3_15g)
def remove_edges(g, edgelist):
    """
    Remove the edges in edgelist from the graph g.

    Arguments:
    g -- undirected graph
    edgelist - list of edges in g to remove

    Returns:
    None
    """
    for edge in edgelist:
        (u, v) = tuple(edge)
        g[u].remove(v)
        g[v].remove(u)        

def gn_graph_partition(g):
    """
    Partition the graph g using the Girvan-Newman method.

    Requires connected_components, shortest_path_edge_betweenness, and
    compute_q to be defined.  This function assumes/requires these
    functions to return the values specified in the homework handout.

    Arguments:
    g -- undirected graph

    Returns:
    A list of tuples where each tuple contains a Q value and a list of
    connected components.
    """
    ### Start with initial graph
    c = connected_components(g)
    q = compute_q(g, c)
    partitions = [(q, c)]

    ### Copy graph so we can partition it without destroying original
    newg = comp182.copy_graph(g)

    ### Iterate until there are no remaining edges in the graph
    while True:
        ### Compute betweenness on the current graph
        btwn = shortest_path_edge_betweenness(newg)
        #print "something is left"
        #print "between value", btwn
        if not btwn:
            #print "between value:", btwn
            ### No information was computed, we're done
            break

        ### Find all the edges with maximum betweenness and remove them
        maxbtwn = max(btwn.values())
        maxedges = [edge for edge, b in btwn.iteritems() if b == maxbtwn]
        remove_edges(newg, maxedges)

        ### Compute the new list of connected components
        c = connected_components(newg)
        #print c
        if len(c) > len(partitions[-1][1]):
            ### This is a new partitioning, compute Q and add it to
            ### the list of partitions.
            q = compute_q(g, c)
            partitions.append((q, c))

    return partitions

def testrungngraphpart():
    """Tests to see if everything is correct with the gn graph partitions function using the
    function testutils.

    Arguments:
    None

    Returns:
    True if the test passes and falses if the test fails."""
    
    data2 = gn_graph_partition(bookgraphs.fig3_15g)
    data3 = [(2.2204460492503131e-16, [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]), (0.2770083102493075, [set([1, 2, 3, 4, 5]), set([6]), set([8, 9, 10, 11, 7])]), (-0.39889196675900257, [set([1]), set([2]), set([3]), set([4]), set([5]), set([6]), set([7]), set([8]), set([9]), set([10]), set([11])])]
    print hw3_testutil.test_partitions_compare(data2,data3,10e-6)

### Use the following function to read the
### 'rice-facebook-undergrads.txt' file and turn it into an attribute
### dictionary.

def read_attributes(filename):
    """
    Code to read student attributes from the file named filename.
    
    The attribute file should consist of one line per student, where
    each line is composed of student, college, year, major.  These are
    all anonymized, so each field is a number.  The student number
    corresponds to the node identifier in the Rice Facebook graph.

    Arguments:
    filename -- name of file storing the attributes

    Returns:
    A dictionary with the student numbers as keys, and a dictionary of
    attributes as values.  Each attribute dictionary contains
    'college', 'year', and 'major' as keys with the obvious associated
    values.
    """
    attributes = {}
    with open(filename) as f:
        for line in f:
            # Split line into student, college, year, major
            fields = line.split()
            student = int(fields[0])
            college = int(fields[1])
            year    = int(fields[2])
            major   = int(fields[3])
            
             # Store student in the dictionary
            attributes[student] = {'college': college,
                                   'year': year,
                                   'major': major}
    return attributes

def thekarategraph():
    """runs the gn function and plots the data to a graph.

    Arguments:
    None

    Returns:
    A pictorial graph from the results of the gn run.

    """

    data = gn_graph_partition(bookgraphs.fig3_13g)
    
    plottingdic = {}
    counter = 0

    for item in data:
        plottingdic[counter] = item[0]
        counter += 1
    print plottingdic

    comp182.plot_dist_linear(plottingdic, "The karate graph", "Number of partitions", "Q value", "The Karate Graph")
    (q, com) = max(data)
    print ""
    print "Maximum Q Value:", q
    print ""
    print len(com), "communities"
    print ""
    print com
    
def thefacebookgraph():
    """Runs the facebook stuff.

    Arguments:
    None

    Returns:
    A pictorial graph pertaining to the facebook data"""

    graph = read_attributes('rice-facebook-undergrads.txt')
    graph1 = comp182.read_graph('rice-facebook.repr')
    data1 = gn_graph_partition(graph1)
    plottingdic = {}
    maxcount = []
    counter = 0
    for item in data1:
        plottingdic[counter] = item[0]
        counter += 1
    #print plottingdic

    comp182.plot_dist_linear(plottingdic, "The Facebook Graph", "Number of Partitions", "Q Value", "The Facebook Graph")
    #print max(graph.values())
    for com in data1:
        maxdic = {'college': collections.defaultdict(int), 'year': collections.defaultdict(int), 'major': collections.defaultdict(int)}
        #print com, "community"
        #print ""
        for stud in com[1]:
            for sub in stud:
                #print sub, "student"
                maxdic['college'][graph[sub]['college']] += 1
                #print maxdic['college'][graph[sub]['college']]
                maxdic['year'][graph[sub]['year']] += 1
                #print "the problem node", graph[sub]['major']
                maxdic['major'][graph[sub]['major']] += 1
        maxcount.append(maxdic)
    print "Highest attributes:", max(maxcount)
    (q, com) = max(data1)
    print ""
    print "Maximum Q Value:", q
    print ""
    print len(com), "communities"
    print ""
    print com
    

def lookatgraph():
    """Function helped to look inside the given facebook file
    and to look at properties given.

    Arguments:
    None

    Returns:
    Shares subsets of the facebook graph."""

    graph = read_attributes('rice-facebook-undergrads.txt')
    counter = 0

   
    for item in graph:
        if counter == 9:
            break
        print "the item", item
        print ""
        #for stuff in graph[item]:
        print graph[item]['major']
        counter += 1

def findgraphquailites(g):
    """Uses the function gn_graph_partition to find the highest Q and community
    and prints them to the screen.

    Arguments:
    g - undirected graph

    Returns:
    None.
    """
    
    (q, com) = max(gn_graph_partition(g))
    print "Maximum Q Value:", q
    print ""
    print len(com), "communities"
    print ""
    print com

    

    return
