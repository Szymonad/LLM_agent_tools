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


def describe_table(table):
    sql = (
        "SELECT column_name, data_type, nullable FROM user_tab_columns "
        "WHERE table_name = UPPER(:table_name) ORDER BY column_id"
    )
    with connect() as connection:
        with connection.cursor() as cursor:
            # The table name comes from the model, so it travels as a bind variable.
            cursor.execute(sql, table_name=table)
            rows = cursor.fetchall()

    if not rows:
        return f"Table {table} does not exist. Call list_tables to see the available tables."
    # Short "NAME TYPE" lines cost far fewer tokens than a list of dictionaries.
    return [f"{name} {data_type}" + ("" if nullable == "Y" else " NOT NULL") for name, data_type, nullable in rows]


def run_query(sql):
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()

    if not rows:
        return "The query returned no rows."
    return {"columns": columns, "rows": rows}
