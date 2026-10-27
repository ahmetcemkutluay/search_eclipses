## subtasks
from tess_products import product

import time
from pathlib import Path

from multiprocessing import Pool #multiprocess your analysis to speed up


##### TASK-1 - open the files and set configurations.

rec_path = Path('..') / 'platolops2_catalog' / 'records_v8.parquet'
meas_path = Path('..') / 'platolops2_catalog' / 'measurements.parquet'
fig_out_path = Path('..') / 'figures'
file_out_path = Path('.') / 'output'

products = product(rec_path, meas_path, file_out_path, fig_out_path)
tics = products.get_tics()

t0 = time.perf_counter()
obs = products.query(tics=tics[:]) #make query for all 
dt = time.perf_counter() - t0   