"""Feature selection registry (the Pipeline's "select" step)."""

_SELECTORS = {
    "passthrough": lambda cfg: "passthrough",
    # TODO: add real selectors
}


def build_selector(cfg: dict):
    """The selector named in `cfg["selection"]["type"]`."""
    selector_type = cfg["selection"]["type"]
    if selector_type not in _SELECTORS:
        raise NotImplementedError(f"TODO: selector '{selector_type}' not implemented. Options: {list(_SELECTORS)}")
    return _SELECTORS[selector_type](cfg)
