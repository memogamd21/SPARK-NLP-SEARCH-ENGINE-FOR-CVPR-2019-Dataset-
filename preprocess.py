import re
import string

def lesseqtwlttrs(word):
    return len(word) <= 2

def rplcdig(st):
    return re.sub('\d', '', st)

def preprocess(X):
    X = X.split()
    X = [''.join(c for c in s if c not in string.punctuation) for s in X]
    X = map(rplcdig, X)
    X = [s for s in X if s]
    X = [s for s in X if not lesseqtwlttrs(s)]
    X = [x.lower() for x in X]
    return X
