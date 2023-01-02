import os

user_to_repos = {}
base_dir = os.getcwd()
import sqlite3


def read_res(con):
    cur = con.cursor()

    insert_str = """
        INSERT INTO simulateExp VALUES
            ({}, {}, "{}", {}) 
    """.format(1, 2, "a b c d ef g", "1234.13")
    cur.execute(insert_str)

    res = cur.execute("SELECT * FROM simulateExp WHERE run_num = {} and model_explored = {}".format(1, 2))
    fetch_res = res.fetchone()
    print(fetch_res)
    # print(json.loads(fetch_res[0])[-5:])


tfmem_smt_file = os.path.join(
    base_dir,
    "result_base/result_system/simulate/TFMEM_201_200run_3km_ea_DB")

con = sqlite3.connect(tfmem_smt_file)

try:
    con.execute("CREATE TABLE simulateExp(run_num, model_explored, top200_model_list, current_x_time)")
except:
    print("already exist")

##create index
# con.execute("CREATE INDEX index_name on simulateExp (run_num, model_explored);")

cur = con.cursor()

res = cur.execute("SELECT count(*) FROM simulateExp")
print(res.fetchall())
# read_res(con)
