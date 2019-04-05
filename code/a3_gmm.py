from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))

def vec_logbm(m,X,myTheta):
    """
    input:
        m -> Scalar
        X -> Txd matrix

    output:
        result -> 1xT matrix
    """
    d = X.shape[1]
    result = -np.divide(np.square(X-myTheta.mu[m]),2*myTheta.Sigma[m]).sum(axis = 1) - (d/2)*np.log(2*np.pi) - 0.5*np.log(myTheta.Sigma[m].prod())
    result = result.reshape(1,X.shape[0])
    
#    result = -np.divide(np.square(X-myTheta.mu[m]),2*myTheta.Sigma[m]).sum(axis = 1) 
 #   result -= (d/2)*np.log(2*np.pi) - 0.5*np.log(myTheta.Sigma[m].prod())
  #  result = result.reshape(1,X.shape[0]) 
    assert(result.shape == (1,X.shape[0])), f"result has shape {result.shape}"
   
    return result

def vec_logpm(m,X,myTheta):
    """
    input:
        m -> scalar
        X -> T x d 
    result = 1 by T matrix
    """

    M = myTheta.omega.shape[0]
    T = X.shape[0]
    logBs = np.zeros((M,T))

    for i in range(M):
        logBs[i] = vec_logbm(i,X,myTheta) #m x T
     
       
    denom = logsumexp(logBs, axis=0, b= myTheta.omega)
    result = (logBs[m] + np.log(myTheta.omega[m])) - denom
    result = result.reshape(1,X.shape[0])
    assert(result.shape == (1,X.shape[0])), f"result has shape {result.shape}"
    assert(np.max(result) <= 0)
    return result

    


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
   
    d = float(len(x))
    result = -np.divide(np.square(x-myTheta.mu[m]),2*myTheta.Sigma[m]).sum() - (d/2)*np.log(2*np.pi) - 0.5*np.log(myTheta.Sigma[m].prod())

    return result  

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M = myTheta.omega.shape[0]
    logbm = log_b_m_x(m,x,myTheta)
    
    result = np.log(myTheta.omega[m][0]) + logbm
    bs = np.ones((M,1))
    for i in range(M):
        bs[i] = log_b_m_x(i,x,myTheta)
    denom = logsumexp(bs, b=myTheta.omega)
    #assert (denom == logsumexp([log_b_m_x(i,x,myTheta) for i in range(M)], b = myTheta.omega)), f"{denom} vs {logsumexp([log_b_m_x(i,x,myTheta) for i in range(M)], b = myTheta.omega)}"
    #result -= logsumexp([log_b_m_x(i,x,myTheta) for i in range(M)], b = myTheta.omega)
    result = result - denom
    return result


    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
   
    logP = np.sum(logsumexp(log_Bs, axis = 0, b=myTheta.omega))
    return logP

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    T = X.shape[0]
    d = X.shape[1]
    #initializing theta
    myTheta = theta( speaker, M, d )
    myTheta.mu = np.random.permutation(X)[:][:M]
    myTheta.Sigma = np.ones(myTheta.Sigma.shape)
    myTheta.omega = np.random.rand(M,1)
    myTheta.omega /= np.sum(myTheta.omega) 

    i = 0 
    prev_L = np.NINF
    improvement = np.inf
    
    logBs = np.ones((M, T))  #eq1
    logPs = np.ones((M, T))  #eq2
    
    f = open('train.txt', 'a+')
    f.write(f"Speaker: {speaker}\n")

    do_vec = True
    while i <= maxIter and improvement > epsilon:
        print(i)
        
        vecBs = np.ones((M,T))
        vecPs = np.ones((M,T))
        for m in range(M):  
            #vecBs[m] = vec_logbm(m,X,myTheta)
            #vecPs[m] = vec_logpm(m,X,myTheta)
            #print(m)
            #for t in range(T):
             #   logBs[m][t] = log_b_m_x(m,X[t],myTheta)
            #    logPs[m][t] = log_p_m_x(m,X[t],myTheta)	
             #   assert(vecBs[m][t] == logBs[m][t])
            #    assert(vecPs[m][t] == logPs[m][t]), f"{vecPs[m][t]} vs {logPs[m][t]}"
            
            #assert(max(logBs[m]) <= 0), f"Invalid probability value for logb {max(logBs[m])}"
            #assert(max(logPs[m]) <= 0), "Invalid probability value for logp"
            #assert(np.nan not in np.exp(logBs[m]))
            #assert(np.nan not in np.exp(logPs[m]))
            logBs[m] = vec_logbm(m,X,myTheta)
            logPs[m] = vec_logpm(m,X,myTheta)
        assert (logBs.shape == (M,T)), "bad bs"
        

        L = logLik(logBs, myTheta)
        print(L)
        f.write(f"Iter: {i}, LogLik: {L}\n")
        omegaHat = np.exp(logsumexp(logPs, axis=1))/T
        
        assert(not np.isnan(np.sum(omegaHat))), "nan in omegaHat"
        
        omegaHat = omegaHat.reshape((M,1))
        assert (omegaHat.shape == myTheta.omega.shape), f"bad omega calculation: shape w_hat: {omegaHat.shape}, shape omega: {myTheta.omega.shape}"

        muHat = np.dot(np.exp(logPs),X)/np.exp(logPs).sum(axis=1).reshape((M,1))
        muHat = muHat.reshape((M,d)) 
        #testmuHat = np.zeros(myTheta.mu.shape)

        #for m in range(M):
        #    testmuHat[m] = np.zeros((1,d))
        #    for t in range(T):
        #        testmuHat[m] += np.exp(logPs[m][t])*X[t]
            
        #    assert(muHat[m].all() == testmuHat[m].all()), f"{muHat[m]} vs {testmuHat[m]}"
        #testmuHat = np.divide(testmuHat, np.exp(logPs).sum(axis=1).reshape((M,1)))
       
        #testmuHat = testmuHat.reshape((M,d))   

        #assert (testmuHat.all() == muHat.all())
        assert (muHat.shape == myTheta.mu.shape), "bad mu"

        sigmaHat = np.dot(np.exp(logPs), np.square(X))/np.exp(logPs).sum(axis=1).reshape((M,1)) - np.square(muHat)
        sigmaHat = sigmaHat.reshape((M,d))

        #testsigmaHat = np.zeros(myTheta.Sigma.shape)      
        #for m in range(M):
        #    testsigmaHat[m] = np.zeros((1,d))
        #    for t in range(T):
        #        testsigmaHat[m] += np.exp(logPs[m][t])*np.square(X[t])
        
        #    testsigmaHat[m] = np.divide(testsigmaHat[m], np.exp(logPs[m]).sum()) - np.square(muHat[m])
        #    assert (testsigmaHat[m].all() == sigmaHat[m].all()), f'{testsigmaHat[m]} vs {sigmaHat[m]}'
        #testsigmaHat = testsigmaHat.reshape((M,d))   

        sigmaHat = sigmaHat.reshape((M,d))
        #assert(testsigmaHat.all() == sigmaHat.all())
        assert (sigmaHat.shape == myTheta.Sigma.shape), "bad sigma"

        myTheta.mu = muHat
        myTheta.omega = omegaHat
        myTheta.Sigma = sigmaHat
         
        improvement = L - prev_L
        #print(improvement) 
        prev_L = L
        i += 1

      
    #print ('TODO')
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    ''' 
    
    bestModel = -1
    d = np.shape(mfcc)[1]
    T = np.shape(mfcc)[0]
    f = open("gmmLiks.txt", "a")
    logBs = np.zeros((M,T))
    logPs = np.zeros((M,T))
   
    Ls = []
    for theta in models:
        for m in range(M):
            logBs[m] = vec_logbm(m,mfcc,theta)
        L = logLik(logBs, theta)
        Ls.append(L)

    sortedInds = reversed(np.argsort(Ls))
    sortedInds = [j for j in sortedInds]

    bestModel = sortedInds[0]
    if k>0:
        f.write(f"Actual ID: {correctID}\n")
        for i in range(k):
            ind = sortedInds[i]
            f.write(f"{ind} {Ls[ind]}\n")
    
            
    f.close()
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    print("evaluating....")
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)
