import os
import numpy as np
import re

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n+1,m+1))
    B = np.zeros((n+1,m+1))
 
   
    for i in range(1):
        for j in range(m+1):
            R[i][j] = max(i, j)
            if j != 0:
                B[i][j] = 2
    for i in range(n+1):
        for j in range(1):
            R[i][j] = max(i, j)
            if i != 0:
                B[i][j] = 1

    for i in range(1,n+1):
        for j in range(1,m+1):
            de = R[i-1][j]+1 
            sub = R[i-1][j-1] + (not r[i-1]==h[j-1])
            ins = R[i][j-1]+1
            R[i][j] = min(de,sub,ins)
            if R[i][j] == de:
                B[i][j] = 1
            elif R[i][j] == ins:
                B[i][j] = 2
            elif R[i][j] == sub and r[i-1] == h[j-1]:
                B[i][j] = 4
            else:
                B[i][j] = 3
            
    num_subs = 0
    num_dels = 0
    num_ins = 0
    i = n
    j = m
    while B[i][j] != 0:
        if B[i][j] == 1:
            num_dels += 1
            i-=1

        elif B[i][j] == 2:
            num_ins +=1
            j-=1

        elif B[i][j] == 3:
            num_subs += 1
            i -=1
            j -=1

        elif B[i][j] == 4:
            i -=1
            j -=1

        else:
            print("Something bad happened")

    
    WER = R[n][m]/n if n != 0 else np.inf
   

    return WER, num_subs, num_ins, num_dels


if __name__ == "__main__":
    assert (Levenshtein("who is there".split(), "".split()) == (1.0,0,0,3)), Levenshtein("who is there".split(), "".split())
    assert (Levenshtein("".split(), "who is there".split()) == (np.inf, 0, 3,0)), Levenshtein("".split(), "who is there".split())
    
    puncs = r'([!"#$%&\\()*+,-/:;<=>?@^_`{|}~])'
    fh = open('lev.txt', 'w+')
     
    mem_g = []
    mem_k = []
    for root,dirs,files in os.walk(dataDir):
        for speaker in dirs:
            with open(f"{root}{speaker}/transcripts.txt", 'r') as f:
                ref_lines = f.readlines()

            with open(f"{root}{speaker}/transcripts.Google.txt", 'r') as f:
                goog_lines = f.readlines()

            with open(f"{root}{speaker}/transcripts.Kaldi.txt", 'r') as f:
                kald_lines = f.readlines()        

            for i in range(len(ref_lines)):
                ref = re.sub(puncs, '', ref_lines[i]).lower().split()[2:]
                goog = re.sub(puncs, '', goog_lines[i]).lower().split()[2:]
                kald = re.sub(puncs, '', kald_lines[i]).lower().split()[2:]


                g_score = Levenshtein(ref, goog)
                k_score = Levenshtein(ref, kald) 
                mem_g.append(g_score)
                mem_k.append(k_score)
                g_print = f"{speaker} Google {i} {g_score[0]} S: {g_score[1]}, I: {g_score[2]}, D: {g_score[3]}\n"
                k_print = f"{speaker} Kaldi {i} {k_score[0]} S: {k_score[1]}, I: {k_score[2]}, D: {k_score[3]}\n"
                fh.write(g_print)
                fh.write(k_print)
                print(g_print)
                print(k_print)
    means_g = np.mean(mem_g, axis =0)
    means_k = np.mean(mem_k, axis =0)
    sig_g = np.var(mem_g, axis =0)
    sig_k = np.var(mem_k, axis =0)

    anal = f"G:: Mean MER = {means_g[0]}, Var MER = {sig_g[0]}\nK:: Mean Mer = {means_k[0]}, Var MER = {sig_k[0]}"
    fh.write(anal)
