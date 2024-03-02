import logging

import sys
import json


import os
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory


from diskcache import Cache

logger = logging.getLogger(__name__)

# These are internal propreitary libraries to a company, they just provide a postgres connection handle and an initialized langchain model
from db import db
from ai_tools import lib_model

lib_model.init()


import langdspy

logger = logging.getLogger(__name__)

_cache = Cache(".sql_cache")
def is_sql(input, output_val, kwargs) -> bool:
    return "```sql" not in output_val

class BaseSqlPrompt(langdspy.PromptSignature):
    hint_sql = langdspy.HintField(desc="Answer in valid PostgreSQL format, do not use markdown formatting.")
    hint_executable = langdspy.HintField(desc="SQL will be executed using psycopgq python extension and should not use slash commands such as '\dt'")
    hint_slash_commands = langdspy.HintField(desc="Slash commands such as \dt only work via the command line and not in the context of a SQL prompt.")

class GenerateSQLForTableList(BaseSqlPrompt):
    hint = langdspy.HintField(desc="Generate PostgreSQL to list all tables in the database schema.")

    # Output the SQL command to list tables
    sql_command = langdspy.OutputField(name="PostgreSQL Command", desc="PostgreSQL command to list all tables as a plaintext executable sql command", validator=is_sql)

class ListDatabaseTables(langdspy.Model):
    generate_sql = langdspy.PromptRunner(template_class=GenerateSQLForTableList, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input, config):
        conn = input['conn']
        sql_res = self.generate_sql.invoke({}, config=config)
        logger.debug(sql_res.sql_command)
        query_results = db.execute_query(sql_res.sql_command, conn)
        # Assume execute_query returns a list of table names
        return list(query_results)


class GenerateSQLForSchemaDescription(BaseSqlPrompt):
    table_name = langdspy.InputField(name="Table Name", desc="The name of the table to describe")

    sql_command = langdspy.OutputField(name="SQL Command", desc="SQL command to describe the table's schema, should reference the schema tables not using slash commands.")

class DescribeTableSchema(langdspy.Model):
    generate_sql = langdspy.PromptRunner(template_class=GenerateSQLForSchemaDescription, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input, config):
        conn = input['conn']
        table_name = input['table_name']
        sql_res = self.generate_sql.invoke({'table_name': table_name}, config=config)
        query_results = db.execute_query(sql_res.sql_command, conn)
        # Parse and format the schema description from query results
        return list(query_results)

class GenerateSQLForRelationships(BaseSqlPrompt):
    table_name = langdspy.InputField(name="Table Name", desc="The name of the table to identify relationships.")

    sql_command = langdspy.OutputField(name="PostgreSQL Command", desc="PostgreSQL command to identify relationships.")

class IdentifyTableRelationships(langdspy.Model):
    generate_sql = langdspy.PromptRunner(template_class=GenerateSQLForRelationships, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input, config):
        conn = input['conn']
        sql_res = self.generate_sql.invoke({'table_name': input['table_name']}, config=config)
        query_results = db.execute_query(sql_res.sql_command, conn)
        # Parse and format the relationships description from query results
        return list(query_results)


def load_schema_map(conn):
    def get_table_details(table_name, conn, config):
        # Attempt to retrieve cached details
        cached_details = _cache.get(f"{table_name}_details")
        if cached_details:
            logger.debug(f"Using cached details for table {table_name}")
            return cached_details

        # input(f"Hit enter to continue with table {table_name}")

        describe_table_model = DescribeTableSchema()
        identify_relationships_model = IdentifyTableRelationships()

        # Describe table schema
        schema_description = describe_table_model.invoke({'table_name': table_name, 'conn': conn}, config=config)
        logger.debug(schema_description)

        # Identify relationships (if applicable)
        relationships_description = identify_relationships_model.invoke({'table_name': table_name, 'conn': conn}, config=config)
        logger.debug(relationships_description)

        # Cache the table details only if the schema_description is not empty
        if schema_description:
            _cache.set(f"{table_name}_details", {'Schema': schema_description, 'Relationships': relationships_description})

        res = {'Schema': schema_description, 'Relationships': relationships_description}

        logger.debug(res)

        # input(f"Hit enter to continue with table {table_name} (finish)")
        
        return res

    def _load_schema_map():
        list_tables_model = ListDatabaseTables()

        config = {'llm': lib_model.get_smart_llm()}
        tables_list = list(list_tables_model.invoke({'conn': conn}, config=config))

        logger.debug(tables_list)
        
        schema_map = {}
        
        # Describe schema for each table and identify relationships
        for table_name, in tables_list:
            table_details = get_table_details(table_name, conn, config)
            schema_map[table_name] = table_details
            if not table_details['Schema']:
                logger.debug(f"No schema description for table {table_name}")
                continue

        return schema_map

    schema_map = _load_schema_map()

    return schema_map

def format_and_print_schema_map(schema_map):
    # Start with an empty string that will accumulate the formatted schema description
    formatted_schema = ""
    
    # Iterate over each table in the schema map
    for table_name, table_info in schema_map.items():
        formatted_schema += f"Table: {table_name}\n"
        formatted_schema += "  Columns:\n"
        
        # Assuming table_info['Schema'] contains a list of column descriptions
        for column_description in table_info['Schema']:
            formatted_schema += f"    - {column_description}\n"
        
        # Check if there are relationships and add them to the formatted string
        if 'Relationships' in table_info and table_info['Relationships']:
            formatted_schema += "  Relationships:\n"
            for relationship in table_info['Relationships']:
                formatted_schema += f"    - {relationship}\n"
        
        # Add a separator between tables for clarity
        formatted_schema += "-"*40 + "\n"
    
    # Print the formatted schema map
    print(formatted_schema)



def main():
    conn = db.get_conn()

    history = InMemoryHistory()


    schema_map = load_schema_map(conn)
    print(schema_map)

    # Assuming schema_map is your schema mapping variable
    format_and_print_schema_map(schema_map)


    while True:
        try:
            query = prompt("SQL> ", history=history)
            if query.lower() in ['exit', 'quit']:
                break
            db.execute_query(query, conn)
        except KeyboardInterrupt:
            break
        except EOFError:
            break

    conn.close()

if __name__ == "__main__":
    db.start()
    main()
    db.stop()

