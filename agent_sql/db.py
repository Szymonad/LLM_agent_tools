"""Database side of the SQL agent."""
import os

import oracledb

ORACLE_DSN = "localhost:1521/xepdb1"


def connect():
    return oracledb.connect(
        user=os.environ["ORACLE_USER"],
        password=os.environ["ORACLE_PASSWORD"],
        dsn=ORACLE_DSN,
    )


def list_tables():
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
            return [row[0] for row in cursor.fetchall()]
