import sqlite3
import json
import numpy as np
import numpy.typing as npt

con = sqlite3.connect("test.db")
con.row_factory = sqlite3.Row
cur = con.cursor()


def deserialize_array(text: str) -> npt.NDArray[np.float64]:
    return np.array(json.loads(text), dtype=np.float64)


def deserialize_bipartition(text: str) -> tuple[list[int], ...]:
    return tuple(json.loads(text))


def create_db() -> None:
    cur.execute('''
        CREATE TABLE IF NOT EXISTS network_property(
               idx INTEGER PRIMARY KEY AUTOINCREMENT,
               tpm TEXT,
               tpm_prior TEXT,
               features TEXT
        )
    ''')


def close_db() -> None:
    con.close()


def drop_db() -> None:
    cur.execute('''DROP TABLE IF EXISTS network_property''')


# All non-serialized features are assumed to be floats
def _cast_to_float(obj):
    if isinstance(obj, dict):
        return {k: _cast_to_float(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cast_to_float(v) for v in obj]
    if isinstance(obj, (int, float, np.floating, np.integer, bool)):
        return float(obj)
    if obj is None:
        return None

    return obj


def write_to_db(network_properties) -> None:
    rows = []
    for prop in network_properties:
        rows.append((
            json.dumps(prop["tpm"].tolist()),
            json.dumps(prop["tpm_prior"].tolist()),
            json.dumps(_cast_to_float(prop["features"])),   # <-- cast here
        ))
    with con:
        cur.executemany(
            '''
            INSERT INTO network_property (tpm, tpm_prior, features)
            VALUES (?, ?, ?, ?)
            ''',
            rows
        )
        con.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "tpm": deserialize_array(row["tpm"]),
        "tpm_prior": deserialize_array(row["tpm_prior"]),
        "features": json.loads(row["features"]),
    }


def get_row_by_idx(idx: int) -> dict | None:
    with con:
        res = cur.execute(
            'SELECT tpm, tpm_prior, features FROM network_property WHERE idx = ?',
            [idx]
        )
        row = res.fetchone()

    return _row_to_dict(row) if row is not None else None


def get_all_rows() -> list[dict]:
    with con:
        res = cur.execute('SELECT tpm, tpm_prior, features FROM network_property')
        rows = res.fetchall()

    return [_row_to_dict(row) for row in rows]