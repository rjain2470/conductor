"""
Description: This file executes a validated SQL string against the LMFDB PostgreSQL instance.
It is step 3 in our pipeline.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

_URL = (
    f"postgresql+psycopg2://{os.getenv('LMFDB_USER')}:{os.getenv('LMFDB_PASSWORD')}"
    f"@{os.getenv('LMFDB_HOST')}:{os.getenv('LMFDB_PORT')}/{os.getenv('LMFDB_DBNAME')}"
)

_ENGINE = create_engine(
    _URL,
    connect_args={
        "connect_timeout": 15,
        "options": "-c default_transaction_read_only=on"
    }
)


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query and return the result as a DataFrame."""
    with _ENGINE.connect() as conn:
        return pd.read_sql_query(text(sql), conn)
