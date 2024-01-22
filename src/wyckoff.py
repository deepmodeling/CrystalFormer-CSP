import pandas as pd
import os


df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'wyckoff.csv'))
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
wyckoff_table = np.zeros((len(wyckoff_list), max_len), dtype=int)
for i, sublist in enumerate(wyckoff_list):
    wyckoff_table[i, :len(sublist)] = sublist
wyckoff_table = jnp.array(wyckoff_table)

if __name__=='__main__':

    print (wyckoff_dict[47-1]['1a'])
    print (wyckoff_table[25-1])
    print (wyckoff_table[47-1])
    print (wyckoff_table[99-1])
    print (wyckoff_table[123-1])
    print (wyckoff_table[221-1])
