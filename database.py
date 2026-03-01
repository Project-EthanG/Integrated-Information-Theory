#import iit_computation
#import tpm_generator
import sqlite3
import json

# Store the following in a db: MI, size of tpm, prior, max mi, max part

con = sqlite3.connect("test.db")
cur = con.cursor()

def create_db():
    cur.execute('''CREATE TABLE IF NOT EXISTS network_property
                   (
                       idx
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       ii
                       REAL,
                       mi
                       REAL,
                       max_bipartition
                       TEXT,
                       max_mi
                       REAL,
                       num_nodes
                       INTEGER
                   )
                ''')

def close_db():
    con.close()

def drop_db():
    cur.execute('''DROP TABLE IF EXISTS network_property''')

def write_to_db(network_properties: list[tuple[float, float, tuple[list[int]] | None, float, int]]) -> None:
    network_properties_db: list[tuple[float, float, str, float, int]] = []
    for property in network_properties:
        ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes = property
        max_bipartition = json.dumps(max_bipartition)
        network_properties_db.append((ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes))

    with con:
        cur.executemany('''
                        INSERT INTO network_property (ii,
                mi,
                max_bipartition,
               max_mi,
               num_nodes)
                        VALUES (?, ?, ?, ?, ?)''', network_properties_db)
        con.commit()


def get_row_by_idx(idx: int) -> tuple[float, float, tuple[list[int]] | None, float, int] | None:
    with con:
        res = cur.execute('''SELECT ii, mi, max_bipartition, max_mi, num_nodes FROM network_property WHERE idx = ?''', [idx])
        row = res.fetchone()

        if row is None:
            return None

        ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes = row
        max_bipartition = json.loads(max_bipartition)
        return ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes

def get_all_rows():
    cur.execute("SELECT * FROM network_property")
    rows = cur.fetchall()
    return rows
