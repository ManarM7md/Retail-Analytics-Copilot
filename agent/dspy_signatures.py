# dspy_signatures.py
import dspy

class Router(dspy.Signature):
    """Classify the question as 'rag', 'sql', or 'hybrid' based on content."""
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: rag, sql, hybrid")

class NLToSQL(dspy.Signature):
    """Convert natural language question to SQL query using the provided schema."""
    question = dspy.InputField()
    db_schema = dspy.InputField(desc="Database schema information")  # Renamed from 'schema'
    constraints = dspy.InputField(desc="Business constraints from documents")
    sql_query = dspy.OutputField(desc="Valid SQLite query ending with semicolon")

class Synthesizer(dspy.Signature):
    """Generate a typed answer based on query results and context, following the format hint."""
    question = dspy.InputField()
    query_results = dspy.InputField(desc="SQL query results or empty list")
    retrieved_context = dspy.InputField(desc="Retrieved document chunks")
    format_hint = dspy.InputField(desc="Required output format")
    answer = dspy.OutputField(desc="Answer matching the format hint")
    explanation = dspy.OutputField(desc="How answer was derived")
    citations = dspy.OutputField(desc="Comma-separated list of source identifiers")