#import iit_computation
#import tpm_generator
import sqlite3
import json
import numpy as np
import numpy.typing as npt


con = sqlite3.connect("test.db")
con.row_factory = sqlite3.Row
cur = con.cursor()


def deserialize_prior(text: str) -> npt.NDArray[np.float64]:
    return np.array(json.loads(text), dtype=np.float64)


def deserialize_tpm(text: str) -> npt.NDArray[np.float64]:
    return np.array(json.loads(text), dtype=np.float64)


def deserialize_bipartition(text: str) -> tuple[list[int]]:
    return tuple(json.loads(text))


def create_db():
    cur.execute('''CREATE TABLE IF NOT EXISTS network_property
                   (
                       idx INTEGER PRIMARY KEY AUTOINCREMENT,
                       tpm TEXT,
                       ii REAL,
                       mi REAL,
                       max_bipartition TEXT,
                       max_mi REAL,
                       num_nodes INTEGER,
                       tpm_prior TEXT
                   )
                ''')


def close_db():
    con.close()


def drop_db():
    cur.execute('''DROP TABLE IF EXISTS network_property''')


def write_to_db(network_properties: list[tuple[npt.NDArray[np.float64], float, float, tuple[list[int]] | None, float, int, npt.NDArray[np.float64]]]) -> None:
    network_properties_db: list[tuple[str, float, float, str, float, int, str]] = []
    for property in network_properties:
        tpm, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior = property
        # FIX 1: serialize TPM to JSON string
        tpm_serialized = json.dumps(tpm.tolist())
        max_bipartition = json.dumps(max_bipartition)
        tpm_prior = json.dumps(tpm_prior.tolist())
        network_properties_db.append((tpm_serialized, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior))

    with con:
        cur.executemany('''
            INSERT INTO network_property (
                tpm, ii,
                mi,
                max_bipartition,
                max_mi,
                num_nodes,
                tpm_prior)
            VALUES (?, ?, ?, ?, ?, ?, ?)''', network_properties_db)
        con.commit()


def get_row_by_idx(idx: int) -> tuple[npt.NDArray[np.float64], float, float, tuple[list[int]] | None, float, int, npt.NDArray[np.float64]] | None:

    with con:
        res = cur.execute(
            '''
            SELECT tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior
            FROM network_property
            WHERE idx = ?
            ''',
            [idx]
        )

        row = res.fetchone()

        if row is None:
            return None

        tpm, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior = row

        max_bipartition = (
            deserialize_bipartition(max_bipartition)
            if max_bipartition is not None
            else None
        )

        if tpm_prior is None:
            raise ValueError("tpm_prior is NULL in database")
        tpm_prior = deserialize_prior(tpm_prior)

        # FIX 2: deserialize TPM into its own variable
        if tpm is None:
            raise ValueError("tpm is NULL in database")
        tpm = deserialize_tpm(tpm)

        return tpm, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior


def get_all_rows() -> list[
    tuple[
        npt.NDArray[np.float64],
        float,
        float,
        tuple[list[int]] | None,
        float,
        int,
        npt.NDArray[np.float64]
    ]
]:
    with con:
        res = cur.execute(
            """
            SELECT tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior
            FROM network_property
            """
        )
        rows = res.fetchall()

    processed_rows = []

    for row in rows:
        tpm, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior = row

        max_bipartition = (
            deserialize_bipartition(max_bipartition)
            if max_bipartition is not None
            else None
        )

        if tpm_prior is None:
            raise ValueError("tpm_prior is NULL in database")
        tpm_prior = deserialize_prior(tpm_prior)

        # FIX 3: deserialize TPM into its own variable
        if tpm is None:
            raise ValueError("tpm is NULL in database")
        tpm = deserialize_tpm(tpm)

        processed_rows.append(
            (
                tpm,
                float(ii),
                float(mi_Xt_Xtpast),
                max_bipartition,
                float(max_mi),
                int(num_nodes),
                tpm_prior,
            )
        )

    return processed_rows