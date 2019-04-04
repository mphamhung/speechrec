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


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    

    d = np.shape(x)[1]

    to_exp = np.square(np.subtract(x, myTheta.mu[m]))

    to_exp = (-0.5)*np.sum(to_exp/myTheta.Sigma[m])
    
    prod = 1
    for sig in myTheta.Sigma[m]:
        prod *= sig
    
    b = 1/((2*np.pi)**(d/2))*np.sqrt(prod)
    
    out = np.exp(to_exp)*b
    
    return out    

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M = myTheta.omega.shape[0]

    b = np.array([log_b_m_x(i, x, myTheta) for i in range(M)]).reshape((M,1)) 

    num = np.multiply(myTheta.omega[m],b[m])
    assert (np.shape(num) == np.shape([1])), "log p error in num"
    num = num[0]
    
    den = np.sum(np.multiply(myTheta.omega, b))
    
    result = np.log(num/den)

    assert (np.shape(result) == np.shape(1)), "log p error"

    return result


    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    logP = 0
   
   # for t in range(log_Bs.shape[1]):
    #    logP +=logsumexp(log_Bs.T[t],b = myTheta.omega)
    logP = np.sum([logsumexp(log_Bs.T[t],b=myTheta.omega) for t in range(log_Bs.shape[1])])
   
    return logP

    
def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    T = X.shape[0]
    d = X.shape[1]
    #initializing theta
    myTheta = theta( speaker, M, d )
    np.random.seed(0)
    myTheta.mu = np.random.permutation(X)[:][:M]
    myTheta.Sigma = np.ones(np.shape(myTheta.Sigma))/M
    myTheta.omega = np.random.rand(M,1)
    myTheta.omega /= np.sum(myTheta.omega)
    

    i = 0 
    prev_L = np.NINF
    improvement = np.inf
    
    logBs = np.ones((M, T))  #eq1
    logPs = np.ones((M, T))  #eq2

    while i <= maxIter and improvement >= epsilon:
        print(i)
        for m in range(M):
            print("m =",m)
     
            logBs[m] = log_b_m_x(m, X, myTheta)
            logPs[m] = log_p_m_x(m, X, myTheta)

        L = logLik(logBs, myTheta)
        omegaHat = np.exp(logPs).sum(axis=1)/T
        omegaHat = omegaHat.reshape((M,1))
        assert (omegaHat.shape == myTheta.omega.shape), f"bad omega calculation: shape w_hat: {omegaHat.shape}, shape omega: {myTheta.omega.shape}"
        
        muHat = np.dot(np.exp(logPs),X)/np.exp(logPs).sum(axis=1).reshape((M,1))
        muHat = muHat.reshape((M,d))
        assert (muHat.shape == myTheta.mu.shape), "bad mu"
        
        sigmaHat = np.dot(np.exp(logPs), np.multiply(X,X))/np.exp(logPs).sum(axis=1).reshape((M,1)) - np.multiply(muHat,muHat)
        sigmaHat = sigmaHat.reshape((M,d))
        assert (sigmaHat.shape == myTheta.Sigma.shape), "bad sigma"
       
        myTheta.mu = muHat
        myTheta.omega = omegaHat
        myTheta.Sigma = sigmaHat
        
        improvement = L - prev_L
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
    
    logBs = np.zeros((M,T))
    logPs = np.zeros((M,T))
   
    Ls = []
    for theta in models:
        for m in range(M):
            logBs[m] = log_b_m_x(m, X, theta)
            logPs[m] = log_p_m_x(m, X, theta)
        L = logLik(logBs, theta)
        Ls.append(L)

    sortedInds = reversed(np.argsort(Ls))

    if k>0:
        print("Actual ID: ", correctID)
        for i in range(k):
            ind = sortedInds[i]
            print(ind, " ", Ls[ind])
    
            
            
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

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
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)

