# structure_scanner.py
#   tries out many 'hidden layer sizes'
#   to find the smallest possible neural networks
#   with good quality functions
# 
# by https://github.com/drandreaskrueger/
#    started on 2017/02/10

import numpy as np
from mlp_parity import all_parity_pairs, ANN

import time, threading, multiprocessing
import os

RESULTFILE="mlp_parity_results.csv"
INIT=True # False # True # careful: True overwrites old results file (and writes table header)

AV_RANGE=200 # last 200 results are averaged, to be used as quality functions 

def initResultFile(filename=RESULTFILE):
    """
    overwrite perhaps existing results file with new one
    first line: Table headings
    """

    if not INIT: return False
    
    columns=["hidden_layer_sizes", "learning_rate", "epochs", "av costs",
             "av costs last %s" % AV_RANGE,"av errorRate last %s" % AV_RANGE,
             "seconds"]

    resultfile=open(filename, "w")
    resultfile.write("\t".join(columns) + "\n")
    resultfile.close()
    
    print "initialized file: ", filename


def deep(hls=[1024]*2, learning_rate=10e-4, epochs=100, show_fig=True):
    # Challenge - find a deeper, slimmer network to solve the problem
    X, Y = all_parity_pairs(12)
    model = ANN(hls)
    return model.fit(X, Y, learning_rate=learning_rate, print_period=10, epochs=epochs, show_fig=show_fig, silent=True)
   

def deep_withLogging(hls, lr, epochs, endAveragingRange=AV_RANGE, filename=RESULTFILE):
    """
    run the deep() training, and digest the results, and print and write them to file.
    """
    # print "process name=%s  thread name=%s" % (multiprocessing.current_process().name , threading.current_thread().getName())
    # time.sleep(0.5); return 
    
    t=time.time()
    costs,errorRates = deep(hls, lr, epochs, False)
    t=time.time()-t
    
    avCosts=np.mean(costs)
    avCostsEnd=np.mean(costs[-endAveragingRange:])
    avErrorRateEnd=np.mean(errorRates[-endAveragingRange:])            
    
    HLS="%s " % hls
    print
    print HLS + "results:"
    print HLS + "hidden layer sizes =", hls
    print HLS + "learning rate =", lr
    print HLS + "seconds =", t
    print HLS + "average costs = %s . Last %s print periods averaged ..." % (avCosts, endAveragingRange)
    print HLS + " ... av costs = %s" % (avCostsEnd)
    print HLS + " ... av error rate = %s" % (avErrorRateEnd)
    print

    resultfile=open(filename, "a")
    resultfile.write("%s\t%s\t%s" % (hls, lr, epochs) )
    resultfile.write("\t%s\t%s\t%s" % (avCosts, avCostsEnd, avErrorRateEnd) )
    resultfile.write("\t%s\n" % t )
    resultfile.close()

def removeAll(inputList, element):
    """
    removes all appearances of element from inputList 
    """
    while True:
        try:
            inputList.remove(element)
        except:
            break


def structures_list(hls_ranges):
    """
    generate all 'hidden layer sizes' for given ranges.
    discard 0 sizes, discard duplicates.
    print them all.
    Returns list of lists.
    """
    hls_all=[]
    hls=[]
    for L1 in range(hls_ranges[0][0], hls_ranges[0][1]+1):
        for L2 in range(hls_ranges[1][0], hls_ranges[1][1]+1):
            for L3 in range(hls_ranges[2][0], hls_ranges[2][1]+1):
                for L4 in range(hls_ranges[3][0], hls_ranges[3][1]+1):
                    hls=[L1, L2, L3, L4]
                    removeAll(hls, 0)   # remove empty layers
                    # print hls
                    hls_all.append(tuple(hls)) # add this to tasklist
    
    removeAll(hls_all, ()) # remove empty NN
    hls_all = sorted(list(set(hls_all)))  # remove duplicate tuples
    
    print "%d structures: %s" % (len(hls_all), hls_all)
    return map(list, hls_all) # list of tuples to list of lists 


def deep_structure_scanner(hls_maxs, lr, epochs):
    """
    Sequentially run through list of all hidden layer sizes.
    Logs to screen and into results file.
    """
    hls_all = structures_list(hls_maxs)
    
    for hls in hls_all:
        deep_withLogging(hls, lr, epochs)
        print


def deep_structure_scanner_threaded(hls_maxs, lr, epochs, max_threads=1):
    """
    Same as above, but try multithreading.
    Result: Slower not faster! 
    """
    hls_all = structures_list(hls_maxs)
    
    threads=[]
    for hls in hls_all:
        print "threading.activ0e_count() = ", threading.active_count()
        while (max_threads <= threading.active_count()-1): # minus 1 because main thread!
            time.sleep(0.1)        
        #                         deep_withLogging       (hls, lr, epochs)
        t=threading.Thread(target=deep_withLogging, args=(hls, lr, epochs))
        t.start()
        threads.append(t)
        print
        
    for t in threads:
        t.join()
                    
                  
def deep_structure_scanner_multiprocessing(hls_maxs, lr, epochs, max_processes=3):
    """
    Same as above, but with 3 concurrent processes, using a process pool.
    """
    hls_all = structures_list(hls_maxs)

    pool=multiprocessing.Pool(max_processes)
    processes = []
    
    for hls in hls_all:
        p=pool.apply_async(deep_withLogging, (hls, lr, epochs))
        processes.append(p)

    for p in processes:
        p.wait()             
    
    
if __name__ == '__main__':
    os.nice(19) # run with high nice level, so that system stays responsive
    
    initResultFile()

    lr=10e-4
    epochs=200
    hls_ranges=[(1,15), (0,15), (0,15), (0,0)]
    hls_ranges=[(1,15), (16,40), (0,0), (0,0)]
    hls_ranges=[(16,40), (1,40), (0,0), (0,0)]
    
    """
    # serial runs
    t=time.time()
    deep_structure_scanner(hls_ranges, lr, epochs)
    print "total time needed (no threading):", time.time() - t
    exit()
    """
    
    """
    # threaded runs
    t=time.time()
    deep_structure_scanner_threaded(hls_ranges, lr, epochs)
    print "total time needed (threading):", time.time() - t
    exit()
    """
    
    # multiprocessing runs
    t = time.time()
    deep_structure_scanner_multiprocessing(hls_ranges, lr, epochs, 
                                           max_processes=3)
    print "total time needed (multiprocessing):", (time.time() - t)
    
    
    