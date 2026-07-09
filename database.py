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


def create_db() -> None:
    cur.execute('''
        CREATE TABLE IF NOT EXISTS network_property(
               idx INTEGER PRIMARY KEY AUTOINCREMENT,
               tpm TEXT,
               ii REAL,
               mi REAL,
               max_bipartition TEXT,
               max_mi REAL,
               num_nodes INTEGER,
               tpm_prior TEXT,

               clustering_mean REAL,
               clustering_var REAL,
               clustering_max REAL,

               betweenness_mean REAL,
               betweenness_var REAL,
               betweenness_max REAL,

               closeness_mean REAL,
               closeness_var REAL,
               closeness_max REAL,

               pagerank_mean REAL,
               pagerank_var REAL,
               pagerank_max REAL,

               lambda1 REAL,
               spectral_gap REAL,
               spectral_entropy REAL,

               weighted_density REAL,
               weighted_reciprocity REAL,

               num_sccs INTEGER,
               max_scc INTEGER,
               diam INTEGER,

               num_cycles INTEGER,
               mean_cycle REAL,
               max_cycle REAL
            )
        ''')


def close_db() -> None:
    con.close()


def drop_db() -> None:
    cur.execute('''DROP TABLE IF EXISTS network_property''')


def write_to_db(network_properties) -> None:
    network_properties_db = []

    for network_property in network_properties:
        # Perfectly extracts the 7 baseline fields + 23 unrolled nbn_features
        (
            tpm, ii, mi_Xt_Xtpast, max_bipartition, max_mi, num_nodes, tpm_prior,

            clustering_mean, clustering_var, clustering_max,
            betweenness_mean, betweenness_var, betweenness_max,
            closeness_mean, closeness_var, closeness_max,
            pagerank_mean, pagerank_var, pagerank_max,

            lambda1, spectral_gap, spectral_entropy,

            weighted_density,
            weighted_reciprocity,

            num_sccs, max_scc, diam,

            num_cycles, mean_cycle, max_cycle
        ) = network_property

        tpm_serialized = json.dumps(tpm.tolist())
        max_bipartition_serialized = json.dumps(max_bipartition)
        tpm_prior_serialized = json.dumps(tpm_prior.tolist())

        network_properties_db.append((
            tpm_serialized, ii, mi_Xt_Xtpast, max_bipartition_serialized, max_mi, num_nodes, tpm_prior_serialized,

            clustering_mean, clustering_var, clustering_max,
            betweenness_mean, betweenness_var, betweenness_max,
            closeness_mean, closeness_var, closeness_max,
            pagerank_mean, pagerank_var, pagerank_max,

            lambda1, spectral_gap, spectral_entropy,

            weighted_density,
            weighted_reciprocity,

            num_sccs,
            max_scc,
            diam,

            num_cycles,
            mean_cycle,
            max_cycle
        ))

    with con:
        cur.executemany(
            '''
            INSERT INTO network_property (
                tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior,

                clustering_mean, clustering_var, clustering_max,
                betweenness_mean, betweenness_var, betweenness_max,
                closeness_mean, closeness_var, closeness_max,
                pagerank_mean, pagerank_var, pagerank_max,

                lambda1, spectral_gap, spectral_entropy,

                weighted_density, weighted_reciprocity,

                num_sccs, max_scc, diam,

                num_cycles, mean_cycle, max_cycle
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            network_properties_db
        )
        con.commit()


def get_row_by_idx(idx: int) -> list:
    with con:
        res = cur.execute(
            '''
            SELECT
                tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior,

                clustering_mean, clustering_var, clustering_max,
                betweenness_mean, betweenness_var, betweenness_max,
                closeness_mean, closeness_var, closeness_max,
                pagerank_mean, pagerank_var, pagerank_max,

                lambda1, spectral_gap, spectral_entropy,

                weighted_density, weighted_reciprocity,

                num_sccs, max_scc, diam,

                num_cycles, mean_cycle, max_cycle
            FROM network_property
            WHERE idx = ?
            ''',
            [idx]
        )

        row = res.fetchone()

        if row is None:
            return None

        max_bipartition = (
            deserialize_bipartition(row['max_bipartition'])
            if row['max_bipartition'] is not None
            else None
        )

        tpm_prior = deserialize_prior(row['tpm_prior'])
        tpm = deserialize_tpm(row['tpm'])

        return [
            tpm,
            row['ii'],
            row['mi'],
            max_bipartition,
            row['max_mi'],
            row['num_nodes'],
            tpm_prior,

            row['clustering_mean'],
            row['clustering_var'],
            row['clustering_max'],

            row['betweenness_mean'],
            row['betweenness_var'],
            row['betweenness_max'],

            row['closeness_mean'],
            row['closeness_var'],
            row['closeness_max'],

            row['pagerank_mean'],
            row['pagerank_var'],
            row['pagerank_max'],

            row['lambda1'],
            row['spectral_gap'],
            row['spectral_entropy'],

            row['weighted_density'],
            row['weighted_reciprocity'],

            row['num_sccs'],
            row['max_scc'],
            row['diam'],

            row['num_cycles'],
            row['mean_cycle'],
            row['max_cycle']
        ]


def get_all_rows():
    with con:
        res = cur.execute(
            """
            SELECT
                tpm, ii, mi, max_bipartition, max_mi, num_nodes, tpm_prior,

                clustering_mean, clustering_var, clustering_max,
                betweenness_mean, betweenness_var, betweenness_max,
                closeness_mean, closeness_var, closeness_max,
                pagerank_mean, pagerank_var, pagerank_max,

                lambda1, spectral_gap, spectral_entropy,

                weighted_density, weighted_reciprocity,

                num_sccs, max_scc, diam,

                num_cycles, mean_cycle, max_cycle
            FROM network_property
            """
        )
        rows = res.fetchall()

    processed_rows = []

    for row in rows:
        max_bipartition = (
            deserialize_bipartition(row['max_bipartition'])
            if row['max_bipartition'] is not None
            else None
        )

        tpm_prior = deserialize_prior(row['tpm_prior'])
        tpm = deserialize_tpm(row['tpm'])

        processed_rows.append(
            (
                tpm,
                float(row['ii']),
                float(row['mi']),
                max_bipartition,
                float(row['max_mi']),
                int(row['num_nodes']),
                tpm_prior,

                float(row['clustering_mean']),
                float(row['clustering_var']),
                float(row['clustering_max']),

                float(row['betweenness_mean']),
                float(row['betweenness_var']),
                float(row['betweenness_max']),

                float(row['closeness_mean']),
                float(row['closeness_var']),
                float(row['closeness_max']),

                float(row['pagerank_mean']),
                float(row['pagerank_var']),
                float(row['pagerank_max']),

                float(row['lambda1']),
                float(row['spectral_gap']),
                float(row['spectral_entropy']),

                float(row['weighted_density']),
                float(row['weighted_reciprocity']),

                int(row['num_sccs']),
                int(row['max_scc']),
                int(row['diam']),

                int(row['num_cycles']),
                float(row['mean_cycle']),
                int(row['max_cycle'])
            )
        )

    return processed_rows