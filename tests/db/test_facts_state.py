"""Committed facts invalidate serving handles without scanning the corpus."""

from league_predictor import db


def test_generation_tracks_committed_updates_and_rollbacks(tmp_path):
    writer = db.get_connection(tmp_path / "facts.db")
    db.ensure_database_ready(writer)
    reader = db.get_connection(tmp_path / "facts.db")
    before = db.facts_generation(reader)
    writer.execute("INSERT INTO teams (team_id, name) VALUES ('A', 'Alpha')")
    assert db.facts_generation(reader) == before
    writer.commit()
    inserted = db.facts_generation(reader)
    assert inserted > before
    writer.execute("UPDATE teams SET name = 'Bravo' WHERE team_id = 'A'")
    writer.rollback()
    assert db.facts_generation(reader) == inserted
    writer.execute("UPDATE teams SET name = 'Bravo' WHERE team_id = 'A'")
    writer.commit()
    assert db.facts_generation(reader) > inserted
    reader.close()
    writer.close()
