class NoSuchToolError(Exception):
    """Raised when no such tool is found or configured."""


class ToolExecutionError(Exception):
    """Raised when an error occurs in the tool."""
