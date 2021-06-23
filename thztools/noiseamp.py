'''
Noise amp takes in the same inputs as noisevar, sigma, mu & T

1) Variable defining
2) Define the noiseamp function
    Inputs: sigma
            mu
            T
    Outputs: sigmamu
    
3) Get the value of Vmu which is output of noisevar() function
4) square root the resulting matrix
5) Sqrt of Vmu is then Sigmamu
'''

from noisevar import noisevar

mu = [ 1, 2 ,3 ,4, 5, 6, 7, 8, 9, 10, 11]
sigma = [.3, .4 ,.5]
T = .05
def noiseamp(sigma, mu, T):
    sigmamu = np.sqrt(noisevar(sigma, mu, T))  #Are we square rooting the matrix, or the elements individually
    return sigmamu