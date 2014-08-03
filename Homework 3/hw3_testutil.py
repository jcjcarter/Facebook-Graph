"""
How to utilize:

- import this testing module at the top of your solution code or
  wherever you wish to write your tests

- to call the test methods in this module is just like calling methods
  in comp182util

    Ex:
        >>> hw3_testutil.test_fp_compare(0.5, 0.5, 10^e-6)
        True

- for each testing method, there usually exists one parameter for your
  result, generated from running your code, followed by a second
  parameter, 'expected', that your result should hope to match. you
  would probably have to manually construct the 'expected' data
  structures and pass them into the test function.

    Ex:
        # our experimental graph
        g1 = {0: set([1, 3]),
              1: set([0, 2]),
              2: set([1, 3]),
              3: set([0, 2])}
        # run our code
        myBetweenness = shortest_path_edge_betweenness(g1)
        # hardcode our expected result
        expectedBetweenness = {frozenset([1, 2]): 4.0,
                               frozenset([2, 3]): 4.0,
                               frozenset([0, 3]): 4.0,
                               frozenset([0, 1]): 4.0}
        # pass the values through the appropriate test function
        betweennessTestResult = hw3_testutil.test_edge_to_fp_dict_compare(myBetweenness,
                                                                          expectedBetweenness,
                                                                          10e-6)
        # interpret the test result
        if(betweennessTestResult):
            print 'Passed'
        else:
            print 'Failed'
"""    

def test_fp_compare(fpnum, expected, epsilon):
    """
    fpmnum is the experimental floating point number to be compared with expected.

    expected is the desired floating point number.

    epsilon is the amount that fpnum is allowed to diverge from expected.
    ie. expected - epsilon <= fpnum <= expected + epsilon

    Returns a boolean:
        True, if fpnum is within epsilon of expected
        False, otherwise

    Example:
    >>> test_fp_compare(1.00001, 1.00000, .00002)
    True
    >>> test_fp_compare(1.00001, 1.00003, .00001)
    False
    >>>
    """
    #check if within range epsilon
    if (expected - epsilon <= fpnum) and (expected + epsilon >= fpnum):
        return True
    else:
        return False

def test_edge_to_fp_dict_compare(fpdict, expected, epsilon):
    """
    fpdict is a dictionary in which the keys are frozensets
    with two elements that represent an edge in the graph and
    the values are floating point numbers. fpdict is to be
    compared to expected.

    expected is the desired edge to floating point dictionary.

    epsilon is the amount that the values of the fpdict dictionary
    are allowed to diverge from the corresponding values in the expected
    dictionary.

    Returns a boolean:
        True, if fpdict contains all keys of expected, and corresponding
                floating point values within range epsilon
        False, otherwise
    """
    #check if all keys present
    if(test_unordered_list_compare(fpdict.keys(), expected.keys())):
        #iterate through keys and do an fp_compare
        for key in fpdict:
            if(not test_fp_compare(fpdict[key], expected[key], epsilon)):
                return False
        return True
    return False

def test_unordered_list_compare(testlist, expected):
    """
    testlist is a list of elements to be compared to expected.

    expected is the desired list of elements.

    Returns a boolean:
        True, if testlist contains all elements of expected, no more, no less
        False, otherwise
    """
    #copy expected
    expectedCopy = list(expected)
    for element in testlist:
       if(element in expectedCopy):
           #remove corresponding element
           expectedCopy.remove(element)
       else:
           return False
    #check if expected has elements remaining
    if(len(expectedCopy) != 0):
       return False
    return True

def test_partitions_compare(partitions, expected, epsilon):
    """
    partitions is a list of tuples where the first element of the tuple
    is a floating point number, Q, and the second element is a list of
    sets of nodes. partitions is to be compared to expected.

    expected is the desired list of partitions formatted identically.

    epsilon is the amount that the modularity measure, Q, is allowed to
    diverge from the corresponding Q values in the tuples of expected.

    Returns a boolean:
        True, if all tuples of partitions are present in expected, and if
                the Q values of partitions' tuples are within range epsilon
                of corresponding values of Q in expected's tuples, anf if
                their respective sets are matching
        False, otherwise
    """
    #copy expected
    expectedCopy = list(expected)
    #for each Q in partitions, locate one within range in expected
    for tup_p in partitions:
        q_value_p = tup_p[0]
        match_found = False
        for tup_e in expected:
            q_value_e = tup_e[0]
            #do an fp compare
            if(test_fp_compare(q_value_p, q_value_e, epsilon)):
                #found Q within range, now compare sets
                if(test_unordered_list_compare(tup_p[1], tup_e[1])):
                    #matched, cross it out
                    expectedCopy.remove(tup_e)
                    match_found = True
                    break
                else:
                    #missed, but could exist further down the list of tuples
                    continue
        #check if Q was matched at all in expected
        if(not match_found):
            return False
    #check if we matched all elements in expected
    if(len(expectedCopy) == 0):
        return True
    else:
        return False

