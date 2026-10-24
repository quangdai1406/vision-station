"""Repeatability and threshold-selection tools. See FORMAT.md."""


class ToolDependencyError(RuntimeError):
    """numpy / opencv is needed for this tool but is not installed."""
