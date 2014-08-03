import comp182

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
        if not btwn:
            ### No information was computed, we're done
            break

        ### Find all the edges with maximum betweenness and remove them
        maxbtwn = max(btwn.values())
        maxedges = [edge for edge, b in btwn.iteritems() if b == maxbtwn]
        remove_edges(newg, maxedges)

        ### Compute the new list of connected components
        c = connected_components(newg)
        if len(c) > len(partitions[-1][1]):
            ### This is a new partitioning, compute Q and add it to
            ### the list of partitions.
            q = compute_q(g, c)
            partitions.append((q, c))

    return partitions


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
