# a script to check how found_books are stored in the database

import duckdb

db_path = "bookshelf_scanner/data/found_books.duckdb"
conn = duckdb.connect(db_path)

# List all tables
print("Tables in the database:", conn.execute("SHOW TABLES").fetchall())

# Inspect specific tables
table_name = "ocr_results"  
print(f"Contents of {table_name}:")
print(conn.execute(f"SELECT * FROM {table_name}").fetchdf())
