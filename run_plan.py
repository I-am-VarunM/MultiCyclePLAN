# Main script to run the PLAN tool

import argparse
import math
import numpy as np
import operator
import os
import pickle as pk
import re
import subprocess
import sys
import time
import statistics

from datetime import datetime
from itertools import combinations
from scipy.stats.stats import pearsonr
from tqdm import tqdm
from Verilog_VCD import Verilog_VCD as v

################################################################################
# Functions to be modified by user as per the design being analysed.
# Please refer to PLAN.md for the corresponding functions for different designs.
################################################################################

# To read the input values generated during simulation
def loadData():
   a = []
   with open('txtfile', 'r') as f:
       for line in f:
        values = [int(val) for val in line.split()]
        a.append(values)
   #print(a)
   return np.array(a,dtype='int')

# To compute the oracle trace from the the input trace generated and the secret key used during simulation
def computeOracle(k):
    ip1 = loadData()
    data = np.array(ip1)
    y = data^k
    #print(y[0,0])
    #print(y)
    return y

################################################################################
################################################################################

# global variables
vcdpath = 'vcd/'
filepath = 'pkl/'
pairs = []
sigArray1 = {} # stores the value carried by each signal for each run
sigGroup = {}
sigMatrix = {}
cipher = {}
O = {}
togglingSigs = set()
numofclk = 3
# Can be parralelised and this is taking care of time stamp values I think x[0] gives time stamp and
def createClkList(clkList, sname, tv):
    for x in tv:
        #print(x[1])
        if x[0] not in clkList: # if clock is not there in the dict
            clkList[x[0]]=[[],[]]
            clkList[x[0]][0].append(sname)
            clkList[x[0]][1].append(x[1])
        else:
            clkList[x[0]][0].append(sname)
            clkList[x[0]][1].append(x[1])
    return clkList


def readVCD(num_iterations, numofclk): #Can be parallelized but Memory can get full owing to the number of signals
    rng = range(1, num_iterations+1) 
    for name, i in zip([str(x)+'.vcd' for x in rng], rng):
        data = {}
        clockList = {}
        data = v.parse_vcd(vcdpath + name, use_stdout=0)
        #print('Data')
        #print(data)
        for x in data:
            signame = data[x]['nets'][0].get('hier') + '.' + data[x]['nets'][0].get('name')
            #print(signame)
            sigdic = data[x]['tv']
            #print(signame)
            #print(sigdic)
            if len(sigdic) < numofclk:
                    last_value = sigdic[-1]
                    sigdic.extend([last_value] * (numofclk - len(sigdic)))
            #print(type(sigdic))
            updated_list = []
            c =1
            for item in sigdic:
                updated_list.append((c, item[1]))
                c = c+1
            #print(updated_list)
            #print(updated_list)
            #print(data[x]['tv'])
            clockList = createClkList(clockList, signame, list(updated_list))
        #print("New clocklist")
        #print(clockList)
            #print(clockList)
        for m in range(1, numofclk+1):
            with open(filepath + str(i) + '_clk'+str(m)+'.pkl', 'wb') as f:
                pk.dump([m, clockList[m]], f)
        """for m in range(1, numofclk+1):
            with open(filepath + str(i) + '_clk'+str(m)+'.pkl', 'rb') as f:
                print(pk.load(f))"""
        '''with open(filepath + str(i) + '.pkl', 'rb') as f:
            print(pk.load(f))'''
    print('Pickle files have been created successfully...')

def alphaNumOrder(string):
   return ''.join([format(int(x), '05d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])

def initSigArray(rfiles, numofclk):
    vcdname = '1.vcd'
    data = v.parse_vcd(vcdpath + vcdname, use_stdout=0)
    for i in range(1, numofclk+1):
        key = f"c{i}"
        sigArray1[key] ={}
    x = rfiles
    result_dict = {}
    for file_name in x:
        common_part = file_name.split('_')[0]  # Extract the common part (e.g., '1', '2', 'numofclk')
        if common_part not in result_dict:
            result_dict[common_part] = []
        result_dict[common_part].append(file_name)

# Create the final array with common parts and '.pkl' appended
    final_array = [common_part + '.pkl' for common_part in result_dict.keys()]
    #print(final_array)
    for f, n in zip(final_array, range(1, len(final_array) + 1)):
        fname = str(n)
        for i in range(numofclk):
            sigArray1[f"c{i+1}"][fname] = {}   #For a Particular File name we are creating numofclk instances
            for s in data:
                sigArray1[f"c{i+1}"][fname][data[s]['nets'][0].get('hier') + '.' + data[s]['nets'][0].get('name')] = '0'
        #print(fname)
        #print(sigArray1)
        #print("Done")
        #print(fname)
        #print(sigArray1)
    for i in range(1, numofclk+1):
        with open(f'sigArray_{i}.pkl', 'wb') as f:
            for x in sigArray1[f"c{i}"]:
                #print(x)
                pk.dump([x, sigArray1[f"c{i}"][x]], f)
    print("SigArray has been created successfully")

def initpairs(num_iterations):
    return list(combinations(np.linspace(1, num_iterations, num_iterations).astype(int), 2));

def loadSigArray(numofclk):
    for i in range(1,numofclk+1):
        with open(f'sigArray_{i}.pkl', 'rb') as f:
            try:
                while True:
                    temp = []
                    temp = pk.load(f)
                    #print(temp)
                    sigArray1[f"c{i}"].update({temp[0]:temp[1]})
            except EOFError:
                pass

def init(num_iterations, numofclk):
    global pairs, sigs;
    pairs = {}
    loadSigArray(numofclk)
    for i in range(1,numofclk+1):
        pairs[i] = initpairs(num_iterations)
    #print(pairs)
    sigs = {}
    for i in range(1, numofclk+1):
        sigs[f"c{i}"] = [x for x in sigArray1[f"c{i}"]['1']] # All signal names
    #print(sigs)

def updateSigArray(k1, k2, v): #fname = k1, k2 = tempsigs, v = tempvalues
    tempdict = {};
    x = k2[k1]
    y = v[k1]
    filename = k1.split('_')[0]
    clkiter = 'c' + k1.split('_')[1][numofclk:]
    #print(filename)
    #print(clkiter)
    for k, v in zip(x, y):
        sigArray1[clkiter][filename][k] = v
        tempdict[k] = v
    #print(sigArray1)
    return tempdict

# compute Hamming distance between every pair of values for each signal
def HammingDistanceSignalWise(sig):
    tempfile = {}
    #print(pairs)
    for i in range(1,numofclk+1):
        tfile = {}
        for p in pairs[i]:
            temp = []
            p1 = str(p[0])
            p2 = str(p[1])
            s1 = sigArray1[f"c{i}"][p1]
            s2 = sigArray1[f"c{i}"][p2]
            #print("s1")
            #print(s1[sig])
            #print(s2[sig])
            temp.append(bin(int(s1[sig], 2) ^ int(s2[sig], 2)).count('1'))
            tfile[p] = int(np.sum(temp))
        tempfile[f'c{i}'] = tfile
    #print("New Sig")
    #print(sig)
    #print("It's value")
    #print(tempfile)
    return tempfile

def processSignals(sigs):
    print("Printing all the sigs")
    print(sigs)
    for sig in tqdm(sigs, "Processing signals"):
        try:
            #print("hamming distance")
            #print(sig)
            #print(HammingDistanceSignalWise(sig))
            ham = (sig, HammingDistanceSignalWise(sig))
            #print(ham)
            #print("hi")
            temp = {}
            for i in range(1, numofclk+1):
                l = []
                for pair in pairs[i]:
                    x = {}
                    x[f"c{i}"] = ham[1][f"c{i}"]
                    #print(x)
                    l.append(x[f"c{i}"][pair])
                    temp[f"c{i}"]= l
                    #print(ham[0])
                    #print(temp)
                    #print(sig)
                    #print(temp)
                    with open('modules/' + ham[0]+'.pkl', 'ab') as f:
                        pk.dump(temp, f)
            #print(sig)
            #print(temp)
        except Exception as e:
            print("{}:{}".format(sig, e))
            print()
def transformData(signal):
    data = {}
    with open('modules/' + signal + '.pkl', 'rb') as f:
        try:
            while True:
                data[signal] = (pk.load(f))
        except EOFError:
            pass
    return data

def computeAndSaveLeakageScores(leaks_file_path, num_iterations, key_value):
    leaks = {}
    mx = {}
    med ={}
    O = {}
    init(num_iterations, numofclk)
    #print(pairs)
    x = np.empty((len(pairs[1]),), dtype=int)
    y = computeOracle(key_value)
    for i in range(numofclk):
        for j,p in enumerate(pairs[i+1]):
            x[j] = bin(y[p[0]-1][i] ^ y[p[1]-1][i]).count('1') # HD b/w two temp values
        O[i+1] = x
        #print(O)

    for sig in togglingSigs:
        for i in range(1, numofclk+1):
            data = transformData(sig)
            d = data[sig]
            c = d[f'c{i}']
            temp = []
            score = pearsonr(O[i], c)[0]
            if (math.isnan(score)):
                    temp.append(0)
            else:
                    temp.append(np.abs(score))
            leaks[sig] = temp
    print(leaks)

    for m in leaks: # calculate max leakage in each signal for the entire clock cycle
        mx[m] = max(leaks[m])
        med[m] = statistics.median(leaks[m])


    leaks_x = []
    leaks_y = []
    sorted_sigwise = dict(sorted(mx.items(), key=operator.itemgetter(1), reverse=True))

    for x in sorted(mx):
        leaks_x.append(x)
        leaks_y.append(mx[x])

    with open(leaks_file_path, "w") as f:
        f.write("Signal,Leakage\n")
        for x in sorted_sigwise:
            f.write("%s,%.4f\n" %(x, sorted_sigwise[x]))
        f.write("\n")

    return len(sorted_sigwise)

def main(input_file_path, simulation_script, num_iterations, key_value, leaks_file_path, time_file_path, numofclk):
    start_time = time.time()

    # simulation
    #subprocess.run(['./' + simulation_script, input_file_path, str(num_iterations)])

    # analysis
    nc2 = ((num_iterations * (num_iterations - 1)) / 2)
    readVCD(num_iterations, numofclk)
    rfiles = os.listdir(filepath)
    rfiles.sort(key = alphaNumOrder)
    initSigArray(rfiles,numofclk)
    debug = 0  # flag for debugging
    init(num_iterations,numofclk) # mandatory intialisations
    signals = [x for x in sigGroup] # signals present
    for x in signals:
        sigMatrix[x] = []
        for y in range(len(sigGroup[x])):
            temp = []
            sigMatrix[x].append(temp)

    result = []
    #print(len(rfiles))
    #print(rfiles[numofclk])
    for fn in range(1, num_iterations*numofclk + 1):
            tempsigs = {}
            tempvals = {}
            with open(filepath + rfiles[fn-1], 'rb') as file:
                fname = str(rfiles[fn-1].replace('.pkl',''))
                #print(fname)
                temp = pk.load(file)
                tempsigs[rfiles[fn-1].replace('.pkl','')] = temp[1][0]
                #print(tempsigs)
                tempvals[rfiles[fn-1].replace('.pkl','')] = temp[1][1]
                togglingSigs.update(temp[1][0])
                #print(togglingSigs)
                tempdict = updateSigArray(fname, tempsigs, tempvals)

    processSignals(togglingSigs)
    numSigs = computeAndSaveLeakageScores(leaks_file_path, num_iterations, key_value)

    end_time = time.time()

    print("Completed!")

    with open(time_file_path, "w") as sf:
        sf.write("Number of signals: {}\n".format(numSigs))
        sf.write("Total time taken: {:.4f}s\n".format(end_time - start_time))

if __name__ == '__main__':
    # creating the argument parser
    my_parser = argparse.ArgumentParser(description='Pre-silicon power side-channel analysis using PLAN')

    # adding the arguments
    my_parser.add_argument('InputFilePath',
                           metavar='input_file_path',
                           type=str,
                           help='path to the input Verilog file to be analyzed')
    my_parser.add_argument('KeyValue',
                           metavar='key_value',
                           type=int,
                           help='secret value in input Verilog file')
    my_parser.add_argument('SimulationScript',
                           metavar='simulation_script',
                           type=str,
                           help='path to script used for behavioral simulation')
    my_parser.add_argument('Design',
                           metavar='design',
                           type=str,
                           help='name of the design being analysed')
    my_parser.add_argument('-n',
                           '--num-iterations',
                           type=int,
                           action='store',
                           help='number of iterations in behavioral simulation, default value = 1000')
    my_parser.add_argument('-r',
                           '--results-path',
                           type=str,
                           action='store',
                           help='name of directory within results/ directory to store results, default value = current timestamp')

    # parsing the arguments
    args = my_parser.parse_args()

    input_file_path = args.InputFilePath
    key_value = args.KeyValue
    simulation_script = args.SimulationScript
    design = args.Design

    num_iterations = args.num_iterations
    if not num_iterations:
        num_iterations = 1000

    results_path = args.results_path
    if results_path:
        results_path = 'results/' + results_path + '/' + design + '/'
    else:
        results_path = 'results/' + datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '/' + design + '/'

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    leaks_file_path = results_path + "leaks.txt"
    time_file_path = results_path + "time.txt"

    if not os.path.isdir('vcd/'):
        os.makedirs('vcd/')

    if not os.path.isdir('pkl/'):
        os.makedirs('pkl/')

    if not os.path.isdir('modules/'):
        os.makedirs('modules/')

    print("Note: Please check that:")
    print("1. the simulation script ({}) given as argument has the correct line numbers, variable names, max range to generate random values".format(simulation_script))
    print("2. the secret key ({}) given as argument is same as that in the input Verilog file ({}) - please refer to PLAN.md for guidance".format(key_value, input_file_path))
    print("numofclk. this script (run_plan.py) has the correct functions to load data and compute oracle (in the first few lines) - please refer to PLAN.md for guidance")
    print()
    print("If you are sure that the above details are correct, and wish to continue, press Y/y (and enter)")
    print("To stop, press any other key (and enter)")
    user_input = input()
    if user_input == 'y' or user_input == 'Y':
        main(input_file_path, simulation_script, num_iterations, key_value, leaks_file_path, time_file_path,numofclk)