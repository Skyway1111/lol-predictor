"""All-at-once feature assembly oracle for the chunked fit corpus."""

from league_predictor.predraft.features import build_features_batch
from league_predictor.predraft.pipeline.dataset import Dataset, window_queries


def build_matrix(conn, *, start, end, families, indexes):
    queries, labels, meta = window_queries(conn, start, end)
    return Dataset(
        X=build_features_batch(queries, indexes=indexes, families=[f.name for f in families]),
        y=labels,
        meta=meta,
        feature_names=[name for family in families for name in family.feature_names],
        categorical_names=[name for family in families for name in family.categorical_names],
    )
