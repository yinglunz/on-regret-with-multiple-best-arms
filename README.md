# On Regret with Multiple Best Arms
This is the python code for NeurIPS 2020 paper On Regret with Multiple Best Arms. Packages  used include: numpy, pandas, time, multiprocessing, copy, functools, datatime, matplotlib, math and sys. 

652_contest.npy contains means (after preprocessing) of the real-world dataset mentioned in Section 6.3; it is obtained by applying cvs_reader.py on the raw dataset 652_summary_KLUCB.csv. The raw dataset is also available on this [website](https://nextml.github.io/caption-contest-data).

regret_class.py: This contains classes that are imported by other .py files.

regret_wrt_alpha.py: This contains code for Fig.2(a) in Section 6.1. It compare the regret of each algorithm at varying hardness levels. The running time, on a machine with 40 cores (2.00 GHz), is around 12 hours .

regret_curve.py: This code runs with one input argument and it is responsible for generating both Fig.2(b) in Section 6.2 and Fig.3 in Section 6.3. When the input argument is 0: it generate Fig.2(b) and compare regret curves on synthetic dataset; when the input argument is 1: it generates Fig.3 and compare regret curves on real-world dataset. For each task, the running time, on a machine with 40 cores (2.00 GHz), is around 1 hour.

