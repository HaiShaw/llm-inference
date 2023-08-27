import pandas as pd
import numpy as np
import math

def fill(time_ms, type, bs, plen, glen, tp, KINJ, file_name):
    def log2(input):
        return int(math.log2(input))

    df = pd.read_excel(file_name)
    # print(df)

    pivot=df[df == 'BS=' + str(bs)].stack().index.tolist()

    row, col = pivot[0]

    # print("======row", row)
    # print("======colum", col)

    is_KINJ = 0 if KINJ==True else 1
    row_offset = ((log2(glen) - log2(16)) * 10 + log2(tp) * 2) + is_KINJ
    # print("=========row_offset", row_offset)

    target_row = row + row_offset
    # print("=========target_rwo", target_row)

    if type == "prefill":
        col_offset = (log2(plen) - log2(32)) * 5 + 4
    elif type == "decode":
        col_offset = (log2(plen) - log2(32)) * 5 + 5
    # print("=========col_offset", col_offset)

    target_col = df.columns[df.columns.get_loc(col) + col_offset]
    # print("=========target_col", target_col)

    df.at[target_row, target_col] = time_ms

    with pd.ExcelWriter(file_name) as writer:
        df.to_excel(writer, index=False)
