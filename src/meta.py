import pandas as pd
import os

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/wyckoff_list.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
wyckoff_positions = df['Wyckoff Positions'].tolist()

full_string = 'import jax.numpy as jnp\n'
for g, wp_list in enumerate(wyckoff_positions):
    fn_string = 'f_%g = [\n'%(g+1) 
    for wp in wp_list[::-1]:
        fn_string += '          lambda x,y,z : 1.0*jnp.array([%s]),\n'%(wp[0])
    fn_string +='       ]\n\n'
    
    full_string += fn_string

print (full_string) 

full_string= 'fn_dict = {'
for g in range(1, 231):
    full_string += '%g : f_%g,\n'%(g, g)
full_string += '}\n\n'

print (full_string)
