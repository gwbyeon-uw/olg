"""Exception hierarchy for the OLG design framework."""


class OLGError(Exception):
    """Base exception for all OLG errors."""


class DecoderNotInitializedError(OLGError):
    """Raised when decode is attempted before initializing both decoders."""


class NoCompatibleQuartetError(OLGError):
    """Raised when no compatible quartet exists for the current position.

    Attributes:
        position: The quartet position index that failed.
    """

    def __init__(self, position: int, message: str = ""):
        self.position = position
        super().__init__(message or f"No compatible quartet at position {position}")


class DecodingError(OLGError):
    """Raised when decode_all exhausts all retries without producing a valid sequence."""


class FixedPositionError(OLGError):
    """Raised when a fixed-position constraint cannot be satisfied."""


class ConfigValidationError(OLGError):
    """Raised when configuration values are invalid."""
