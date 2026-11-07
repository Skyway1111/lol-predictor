"""Learned region ratings: one walk-forward MAP fit pricing every league pool."""

# Flow: table (what consumers read) -> corpus (vintage-cached games, home
# structure, player matrix) -> fit (fit_region_ratings and its MAP solve).

from league_predictor.predraft.ratings.region_rating.corpus import load_games
from league_predictor.predraft.ratings.region_rating.fit import fit_region_ratings
from league_predictor.predraft.ratings.region_rating.table import RegionRatingTable

__all__ = [
    "RegionRatingTable",
    "fit_region_ratings",
    "load_games",
]
