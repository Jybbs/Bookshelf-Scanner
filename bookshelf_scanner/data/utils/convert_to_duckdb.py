import duckdb
import logging

logging.basicConfig(
    filename = 'json_to_duckdb.log',
    filemode = 'w',
    level    = logging.INFO,
    format   = '%(asctime)s - %(levelname)s - %(message)s'
)

def jsonl_to_duckdb(jsonl_path, duckdb_path, table_name):
    """
    Converts a JSON Lines file to a DuckDB table using DuckDB's native JSON handling.
    
    :param jsonl_path: Path to the input JSONL file.
    :param duckdb_path: Path where the DuckDB database will be created.
    :param table_name: Name of the table to create in DuckDB.
    """
    try:
        # Connect to DuckDB, and create the database file if it doesn't exist
        conn = duckdb.connect(database = duckdb_path, read_only = False)
        logging.info(f"Connected to DuckDB database at '{duckdb_path}'.")
        
        # Create table from JSONL
        create_table_query = f"""
        CREATE TABLE {table_name} AS 
        SELECT * FROM read_json_auto('{jsonl_path}', 
                                     ignore_errors = true, 
                                     union_by_name = true);
        """
        conn.execute(create_table_query)
        logging.info(f"Successfully created table '{table_name}' from '{jsonl_path}'.")
        
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":

    jsonl_to_duckdb(
        jsonl_path  = 'bookMeta.jsonl.json',
        duckdb_path = 'books.duckdb', 
        table_name  = 'books'
    )
