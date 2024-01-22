from pyxtal.symmetry import Group
import pandas as pd


# 创建一个空的DataFrame来存储数据
wyckoff_data = []

# 循环遍历230个空间群
for i in range(1, 231):
    sg = Group(i)
    positions = sg.Wyckoff_positions
    positions.sort(key=lambda x: x.letter, reverse=False) 
    wyckoff_labels = [wp.get_label() for wp in positions]

    # print(positions)
    # print(wyckoff_labels)

    wyckoff_data.append({'Space Group': i, 'Wyckoff Positions': wyckoff_labels})
    # break

wyckoff_df = pd.DataFrame(wyckoff_data)
wyckoff_df.to_csv('./wyckoff.csv', index=False)


### import wyckoff symbols
# from config import *
# from src.wyckoff import wyckoff_symbols

# print(wyckoff_symbols)
