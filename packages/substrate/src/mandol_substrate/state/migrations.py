"""Atomic, forward-only migrations; unknown versions fail closed."""

import sqlite3
import uuid

from .models import StateUnavailable
from .schema import LEXICAL_SCHEMA, PURGE_SCHEMA, RUNTIME_SCHEMA, SCOPE_ATTRIBUTES_SCHEMA, SCHEMA, SCHEMA_VERSION


def migrate(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    if version not in (0, 1, 2, 3, 4):
        raise StateUnavailable(f"unsupported canonical schema version {version}")
    # executescript handles its own BEGIN; never use it inside a business transaction.
    try:
        connection.executescript(
            "BEGIN IMMEDIATE;\n" + (SCHEMA if version == 0 else "")
            + (LEXICAL_SCHEMA if version < 2 else "")
            + (PURGE_SCHEMA if version < 3 else "")
            + (RUNTIME_SCHEMA if version < 4 else "") + SCOPE_ATTRIBUTES_SCHEMA
        )
        if version == 0:
            connection.execute(
                "INSERT INTO manifest VALUES ('db_uuid',?)", (str(uuid.uuid4()),)
            )
        else:
            # No tokenization or automatic corpus rebuild during open. An old
            # populated scope requires explicit lexical repair before serving.
            connection.execute("UPDATE scopes SET schema_version=?", (SCHEMA_VERSION,))
        if version < 2:
            connection.execute("INSERT INTO manifest VALUES ('lexical_schema_version','1')")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
