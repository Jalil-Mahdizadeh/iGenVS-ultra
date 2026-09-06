"""Project-specific exceptions."""


class IGenVSError(RuntimeError):
    """Base class for expected iGenVS runtime failures."""


class InputError(IGenVSError):
    """Raised when an input library or configuration is invalid."""


class ExternalToolError(IGenVSError):
    """Raised when a required external executable fails."""
