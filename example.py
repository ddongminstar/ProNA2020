import pickle
import numpy as np
import json
from prona2019Mod.protvec import to_vecs

word2vec_index = pickle.load(open('word2vec_Pro1.index','rb'))
phrase_model = pickle.load(open('phrase.model','rb'))

sequence = "ATGCATTTTGGGCT"

vector = np.array(to_vecs(sequence, phrase_model, 3, word2vec_index))

print(sequence, vector)
