import json
import sqlite3

# fetch result from simulated result
def fetch_from_db(tf_smt_file, run_id_m, B1_m):
    global con
    global cur
    if con is None:
        con = sqlite3.connect(tf_smt_file)
        cur = con.cursor()

    res = cur.execute(
        "SELECT * FROM simulateExp WHERE run_num = {} and model_explored = {}".format(run_id_m, B1_m))
    fetch_res = res.fetchone()

    arch_id = fetch_res[2]
    candidates = json.loads(fetch_res[3])
    current_time = float(fetch_res[4])

    return arch_id, candidates, current_time