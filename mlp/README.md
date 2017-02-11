# Machine Learning

### inspired by
Deep Learning: Recurrent Neural Networks in Python
https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python and 
https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rnn_class

### install dependencies

```bash
apt-get install python-pip python-tk virtualenv

virtualenv envML
source envML/bin/activate

pip install sklearn nltk theano matplotlib numpy 
```

### run
Change the parameters directly in the code (top and bottom of file). Then run 

    python2 structure_scanner.py
    
It outputs to screen, and writes into:

    mlp_parity_results.csv

### CPU usage
To sleep in the room without the laptop causing fan noises (and make this not a "burn test") I decided to lower the CPU usage:

    cpulimit -p 4282 -l 70 -m

It runs on 1 core. As the runs are independent, it would make sense to parallelize 3 or 4 of the tasks. See below.

Unfortunately, the `cpulimit -m` switch leads to GUI freezing, otherwise it would be easy to limit the main process and all its children:

    ps aux | grep python2 | grep -v "pydev\|grep\|RN+"
    cpulimit -p 4282 -l 70 -m
    
A workaround for multiprocessing is this:

    ps aux | grep python2 | grep -v "pydev\|grep\|SN" | awk '{ system("cpulimit -p " $2 " -l 50 -b " ); }'

It compiles the PIDs of all 3 child processes, and sends them to 3 background `cpulimit -b` each limiting the CPU use to 50%.

### threading and multiprocessing

Test results:

#### epochs=20, hls_maxs=[4,4,0,0]

```
Serial runs, 3 repetitions: 22, 22, 22 seconds total 
 
Threading(max_threads=1) 23 seconds total   
Threading(max_threads=2) 27 seconds total  
Threading(max_threads=3) 30 seconds total     
Threading(max_threads=4) 29 seconds total
  
Multiprocessing(max_processes=1) 26 seconds total  
Multiprocessing(max_processes=2) 15 seconds total  
Multiprocessing(max_processes=3) 14 seconds total  
Multiprocessing(max_processes=4) 13 seconds total  
Multiprocessing(max_processes=5) 31, 22, 21 seconds total (3 repetitions)  
Multiprocessing(max_processes=6) 21, 22, 21 seconds total (3 repetitions)  
```

#### epochs=50, hls_maxs=[5,5,0,0]  

```
Serial runs: 53.4 seconds total  

Multiprocessing(max_processes=2) 34.5 seconds total  
Multiprocessing(max_processes=3) 30.5 seconds total  
Multiprocessing(max_processes=4) 29.1 seconds total  
```

So: Threading makes it slower, multiprocessing makes it faster, within limits.   

