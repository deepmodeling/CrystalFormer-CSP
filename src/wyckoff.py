import pandas as pd
import os


df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/wyckoff_symbols.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list

wyckoff_symbols = df['Wyckoff Positions'].tolist()

import numpy as np
import jax.numpy as jnp

wyckoff_list = []
wyckoff_dict = []
for ws in wyckoff_symbols:
    wyckoff_list.append( [0] +[0 if w == "" else int(w[0]) for w in ws] )

    ws = [""] + ws
    wyckoff_dict.append( {value: index for index, value in enumerate(ws)} )

max_len = max(len(sublist) for sublist in wyckoff_list)
mult_table = np.zeros((len(wyckoff_list), max_len), dtype=int)
for i, sublist in enumerate(wyckoff_list):
    mult_table[i, :len(sublist)] = sublist
mult_table = jnp.array(mult_table)

if __name__=='__main__':

    print (wyckoff_dict[47-1]['1a']) # wyckoff symbol -> ordering 
    print (mult_table[25-1]) # space group id -> multiplicity table
    print (mult_table[47-1])
    print (mult_table[99-1])
    print (mult_table[123-1])
    print (mult_table[221-1])
