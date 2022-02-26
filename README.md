# On Regret with Multiple Best Arms

This repository contains the python code for NeurIPS 2020 paper **On Regret with Multiple Best Arms**. Packages used include: numpy, pandas, time, multiprocessing, copy, functools, datatime, matplotlib, math and sys. 

Use the following commands to reproduce our experiments in Figure 2.

```
python3 regret_wrt_alpha.py
python3 regret_curve.py 0
```

To reproduce our experiment in Figure 3, first obtain `652_summary_KLUCB.csv` from this [website](https://github.com/nextml/caption-contest-data), and then use the following commands.

```
python3 csv_reader.py
python3 regret_curve.py 1
```

On a cluster with 40 cores (2.00 GHz), the runtime for `python3 regret_curve.py 0` (or `python3 regret_curve.py 1`) is around 1 hour, and the runtime for `python3 regret_wrt_alpha.py` is around 12 hours.
