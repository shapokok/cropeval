"""Reading and validating input data.

The input is a CSV file with one row per image and three columns:

- ``y_true``: the true class label,
- ``y_pred``: the label predicted by the model,
- ``domain``: where the image comes from, ``lab`` or ``field``.

Every problem found in the data is reported as a ``DataValidationError``
with a message that says what is wrong and, where possible, in which rows.
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("y_true", "y_pred", "domain")
DOMAINS = ("lab", "field")

# How many offending row numbers to show in an error message.
_MAX_ROWS_IN_MESSAGE = 5


class DataValidationError(ValueError):
    """Raised when the input table does not have the expected format."""


def _format_rows(index: pd.Index) -> str:
    """Return a short human-readable list of CSV line numbers.

    Line numbers are 1-based and count the header as line 1, so they match
    what the user sees when opening the file in an editor.
    """
    lines = [str(i + 2) for i in index[:_MAX_ROWS_IN_MESSAGE]]
    text = ", ".join(lines)
    if len(index) > _MAX_ROWS_IN_MESSAGE:
        text += f" (and {len(index) - _MAX_ROWS_IN_MESSAGE} more)"
    return text


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Check a predictions table and return a cleaned copy.

    Cleaning only normalises the ``domain`` column (strips spaces and
    lower-cases it, so ``" Lab"`` becomes ``"lab"``). Labels are left as is.

    Raises:
        DataValidationError: if columns are missing, the table is empty,
            required values are missing, or ``domain`` has unknown values.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}; "
            f"found: {', '.join(map(str, df.columns)) or 'none'}."
        )

    if len(df) == 0:
        raise DataValidationError("The table has no rows.")

    # Keep only the columns we need and work on a copy, so the caller's
    # DataFrame is never modified.
    df = df[list(REQUIRED_COLUMNS)].copy()

    for col in REQUIRED_COLUMNS:
        empty_rows = df.index[df[col].isna()]
        if len(empty_rows) > 0:
            raise DataValidationError(
                f"Column '{col}' has missing values in CSV line(s) "
                f"{_format_rows(empty_rows)}."
            )

    df["domain"] = df["domain"].astype(str).str.strip().str.lower()
    bad_rows = df.index[~df["domain"].isin(DOMAINS)]
    if len(bad_rows) > 0:
        bad_values = sorted(df.loc[bad_rows, "domain"].unique())
        raise DataValidationError(
            f"Column 'domain' must be one of {', '.join(DOMAINS)}; "
            f"got {', '.join(repr(v) for v in bad_values)} in CSV line(s) "
            f"{_format_rows(bad_rows)}."
        )

    return df


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a predictions CSV file and validate it.

    Raises:
        FileNotFoundError: if the file does not exist.
        DataValidationError: if the file is empty or has the wrong format.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        raise DataValidationError(f"Input file is empty: {path}") from err

    return validate_predictions(df)
