class InvalidResponseError(Exception):
    """Raised when an invalid response is received. E.g. tools to call are not valid, or argument is missing."""


class ProviderFailureError(Exception):
    """Raised when a provider fails to generate a response due to API errors,
    timeouts, or other provider-specific issues."""


class LLMContextOverflowError(ProviderFailureError):
    """Raised when a provider fails to generate a response due to LLM context overflow."""
