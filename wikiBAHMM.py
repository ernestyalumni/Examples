# wikiBAHMM.py
# wikiBAHMM.py - The Wikipedia Bob Alice HMM example using scikit-learn
# cf. http://sujitpal.blogspot.cz/2013/03/the-wikipedia-bob-alice-hmm-example.html
# Shout outs to Sujit Pal with his blog Salmon Run, where he has python implementations of Machine Learning and AI techniques
#
# 20140731
# 
# Fund Science! & Help Ernest finish his Physics Research! : quantum super-A-polynomials - Ernest Yeung
#                                               
# http://igg.me/at/ernestyalumni2014                                                                             
#                                                              
# Facebook     : ernestyalumni  
# github       : ernestyalumni                                                                     
# gmail        : ernestyalumni                                                                     
# linkedin     : ernestyalumni                                                                             
# tumblr       : ernestyalumni                                                               
# twitter      : ernestyalumni                                                             
# youtube      : ernestyalumni                                                                
# indiegogo    : ernestyalumni                                                                        
# 
# Ernest Yeung was supported by Mr. and Mrs. C.W. Yeung, Prof. Robert A. Rosenstone, Michael Drown, Arvid Kingl, Mr. and Mrs. Valerie Cheng, and the Foundation for Polish Sciences, Warsaw University.  
#
#
# This code is open-source, governed by the Creative Common license.  Use of code is governed by the Caltech Honor Code: ``No member of the Caltech community shall take unfair advantage of any other member of the Caltech community.'' 
#

# cf. http://sujitpal.blogspot.cz/2013/03/the-wikipedia-bob-alice-hmm-example.html
#

from __future__ import division
import numpy as np
from sklearn import hmm

states   = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
        [ 0.7, 0.3],
        [ 0.4, 0.6]
])

emission_probability = np.array([
        [ 0.1, 0.4, 0.5],
        [0.6 , 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model._set_startprob(start_probability)
model._set_transmat(transition_probability)
model._set_emissionprob(emission_probability)

# predict a sequence of hidden states based on visible states
bob_says = [ 0, 2, 1, 1, 2, 0]
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print "Bob says:", ", ".join(map(lambda x: observations[x], bob_says))
print "Alice hearts:", ", ".join(map(lambda x: states[x], alice_hears))

