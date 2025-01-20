import os

aurora_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = aurora_dir.rsplit('/aurora', 1)[0] + "/data"
input_dir = aurora_dir.rsplit('/aurora', 1)[0] + "/input"
plots_dir = aurora_dir.rsplit('/aurora', 1)[0] + "/plotting"
runs_dir = aurora_dir.rsplit('/aurora', 1)[0] + "/runs"
