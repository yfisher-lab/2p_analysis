import numpy as np
from time import sleep



def build_cmd_str(queue):
    #These 2 variables can be edited to change bar jump time and closed loop time. Input only as integers.
    #t_bj = 2 #time bar is held in place after jump, in seconds
    t_cl = 30 #time vr system is in closed loop between double jumps, in seconds

    t_ttl = 60*5 #total time is 5 mins for the entire protocol. this is to make data processing easier later
    
    num_trials = int(t_ttl / t_cl) #number of trials (closed loop); int() will round down to nearest integer
    #t_end = t_ttl - (num_trials * ((t_bj)+t_cl)) #time in closed loop at the end of trials. Added if trials do not perfectly fit in 5 minute interval.
    #to make a list of dac bit values equivalent to 0', 90', 180', and 270':
    #max_dac_val = 4096
    #min_dac_val = 0
    #headings = np.arange(0, max_dac_val, max_dac_val, dtype=int)
    # headings = [0,1023,2046,3070]
    
    #closed loop for t_cl seconds
#     queue.put('0,5\n'.encode('UTF-8')) #closed loop
    #sleep(t_cl)
    
    
    for i in range(num_trials):        
        
        #jump the bar by pi (then will just follow the bump)
        queue.put('1,15,3.14\n'.encode('UTF-8'))
        sleep(t_cl)
        
        ##depending on whether holding the bump for some seconds
        #queue.put('0,4\n'.encode('UTF-8')) #open loop  
        #sleep(t_bj)
        #queue.put('0,5\n'.encode('UTF-8')) 
        #sleep(t_cl)

    
#     queue.put('0,5\n'.encode('UTF-8')) #closed loop
#     sleep(t_end)
#     queue.put('1,7,0\n'.encode('UTF-8'))



    




def run(queue):
    
    build_cmd_str(queue)

    return