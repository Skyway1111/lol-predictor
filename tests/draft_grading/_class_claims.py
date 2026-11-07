from league_predictor.draft_grading.champion_attributes import classes


def champion_roles(display_name: str) -> frozenset[str]:
    """Roles the class file clusters ``display_name`` in; empty for CLASS_FILE_GAPS champions."""
    classes.ensure_loaded()
    if display_name in classes.CLASS_FILE_GAPS:
        return frozenset()
    canonical = classes.resolve_form(display_name)
    return frozenset(role for name, role in classes._CHAMP_ROLE_CLASS if name == canonical)
