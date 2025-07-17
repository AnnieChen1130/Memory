"""
Database setup script for the Memory System
"""

import asyncio
import os
import sys

import asyncpg

from src.utils.config import settings


async def setup_database():
    """Initialize the database and run the schema"""
    print("Setting up Memory System database...")
    print(f"Database URL: {settings.database_url}")

    try:
        # Connect to the database
        conn = await asyncpg.connect(settings.database_url)

        # Read and execute the SQL schema
        schema_path = os.path.join(os.path.dirname(__file__), "sql", "init_tables.sql")

        if not os.path.exists(schema_path):
            print(f"Error: Schema file not found at {schema_path}")
            return False

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        print("Executing database schema...")
        await conn.execute(schema_sql)

        print("Database setup completed successfully!")
        await conn.close()
        return True

    except Exception as e:
        print(f"Error setting up database: {e}")
        return False


async def check_database():
    """Check if the database is properly set up"""
    try:
        conn = await asyncpg.connect(settings.database_url)

        # Check if our tables exist
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('memoryitems', 'relationships')
        """)

        table_names = [row["tablename"] for row in tables]

        print(f"Found tables: {table_names}")

        if "memoryitems" in table_names and "relationships" in table_names:
            print("✓ Database is properly set up")

            # Check if pgvector extension is installed
            extensions = await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            )
            if extensions:
                print("✓ pgvector extension is installed")
            else:
                print("⚠ pgvector extension is not installed")
        else:
            print("✗ Database tables are missing")
            return False

        await conn.close()
        return True

    except Exception as e:
        print(f"Error checking database: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print("Checking database setup...")
        success = asyncio.run(check_database())
    else:
        print("Initializing database...")
        success = asyncio.run(setup_database())

    sys.exit(0 if success else 1)
