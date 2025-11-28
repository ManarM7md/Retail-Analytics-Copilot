import sqlite3
from typing import List, Dict, Any, Tuple
import pandas as pd

class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_schema_info(self) -> str:
        """Get schema information for all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema_info = []
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            
            col_info = []
            for col in columns:
                col_info.append(f"{col[1]} ({col[2]})")
            
            schema_info.append(f"Table: {table_name}\nColumns: {', '.join(col_info)}")
        
        conn.close()
        return "\n\n".join(schema_info)
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """Execute SQL query and return results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert to list of dicts
            results = []
            for row in rows:
                results.append({col: val for col, val in zip(columns, row)})
            
            conn.close()
            return results, columns, ""
        except Exception as e:
            conn.close()
            return [], [], str(e)