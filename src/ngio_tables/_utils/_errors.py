class NgioTablesError(Exception):
    """Base class for all errors in the ngio-tables package."""

    pass


class NgioTablesFileNotFoundError(NgioTablesError, FileNotFoundError):
    """Error raised when a file is not found."""

    pass


class NgioTablesFileExistsError(NgioTablesError, FileExistsError):
    """Error raised when a file already exists."""

    pass


class NgioTablesValidationError(NgioTablesError, ValueError):
    """Generic error raised when a file does not pass validation."""

    pass


class NgioTablesTableValidationError(NgioTablesError):
    """Error raised when a table does not pass validation."""

    pass


class NgioTablesValueError(NgioTablesError, ValueError):
    """Error raised when a value does not pass a run time test."""

    pass
