def noiseamp(sigma, mu, T):
    sigmamu = sqrt(noisevar(sigma, mu, T))
    return sigmamu