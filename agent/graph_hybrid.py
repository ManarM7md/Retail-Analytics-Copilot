import ollama
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from .rag.retrieval import SimpleRetriever
from .tools.sqlite_tool import SQLiteTool
import json
import re
import logging
import ast

# Configure logging for our application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  

class AgentState(BaseModel):
    question: str
    classification: str = ""
    retrieved_chunks: List[Dict[str, Any]] = []
    constraints: Dict[str, Any] = {}
    sql_query: str = ""
    sql_results: List[Dict[str, Any]] = []
    sql_columns: List[str] = []
    sql_error: str = ""
    final_answer: Any = None
    explanation: str = ""
    citations: List[str] = []
    confidence: float = 0.0
    repair_count: int = 0
    max_repairs: int = 2
    format_hint: str = ""
    trace: List[str] = []

class RetailAnalyticsCopilot:
    def __init__(self, db_path: str, docs_dir: str):
        self.sqlite_tool = SQLiteTool(db_path)
        self.retriever = SimpleRetriever(docs_dir)
        self.schema_info = self.sqlite_tool.get_schema_info()
        self.model = "phi3.5:3.8b-mini-instruct-q4_K_M"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Generic Ollama LLM call with system and user prompts"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return ""

    def parse_answer(self, answer_str: str, format_hint: str) -> Any:
        """Safely parse answer string to required format based on hint with multiple fallback strategies"""
        if not format_hint or not answer_str:
            return answer_str.strip() if answer_str else answer_str
            
        try:
            answer_str = answer_str.strip()
            format_hint = format_hint.strip().lower()
            
            # Handle special cases first
            if "n/a" in answer_str.lower() or "not available" in answer_str.lower() or "no data" in answer_str.lower():
                if format_hint == "int":
                    return 0
                elif format_hint == "float":
                    return 0.0
                elif format_hint.startswith("{"):
                    return {}
                elif format_hint.startswith("list["):
                    return []
                return answer_str
            
            # Handle numeric formats
            if format_hint == "int":
                # Extract first number from string if needed
                number_match = re.search(r'-?\d+', answer_str)
                if number_match:
                    return int(number_match.group())
                return int(answer_str)
                
            elif format_hint == "float":
                # Extract first floating point number from string if needed
                float_match = re.search(r'-?\d+\.?\d*', answer_str)
                if float_match:
                    return float(float_match.group())
                return float(answer_str)
                
            # Handle structured formats with multiple parsing strategies
            elif format_hint.startswith("{") or format_hint.startswith("list["):
                # Strategy 1: Try direct JSON parsing with quote fixes
                try:
                    fixed_json = self._fix_json_format(answer_str)
                    return json.loads(fixed_json)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Strategy 2: Use ast.literal_eval for Python-like literals
                try:
                    return ast.literal_eval(answer_str)
                except (SyntaxError, ValueError):
                    pass
                
                # Strategy 3: Parse key-value pairs manually for object formats
                if format_hint.startswith("{"):
                    return self._parse_object_format(answer_str, format_hint)
                    
                # Strategy 4: Parse list items manually for list formats
                elif format_hint.startswith("list["):
                    return self._parse_list_format(answer_str, format_hint)
            
            # Fallback: Return cleaned string
            return self._clean_string_value(answer_str)
            
        except Exception as e:
            logger.warning(f"Format parsing failed for '{answer_str}' with hint '{format_hint}': {str(e)}")
            # Best-effort fallback based on format hint
            if format_hint == "int":
                return 0
            elif format_hint == "float":
                return 0.0
            elif format_hint.startswith("{"):
                return {}
            elif format_hint.startswith("list["):
                return []
            return self._clean_string_value(answer_str)

    def _fix_json_format(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses"""
        # Add quotes around unquoted keys
        json_str = re.sub(r'(\s*)([^"\s{}:,]+)(\s*):', r'\1"\2"\3:', json_str)
        
        # Convert single quotes to double quotes (carefully)
        json_str = json_str.replace("'", '"')
        
        # Fix common boolean values
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        # Handle N/A values
        json_str = re.sub(r'"?n/a"?', 'null', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'"?not available"?', 'null', json_str, flags=re.IGNORECASE)
        
        # Ensure proper structure
        if json_str.startswith("[") and not json_str.endswith("]"):
            json_str = json_str + "]"
        elif json_str.startswith("{") and not json_str.endswith("}"):
            json_str = json_str + "}"
            
        return json_str

    def _parse_object_format(self, answer_str: str, format_hint: str) -> Dict[str, Any]:
        """Parse object format with flexible key-value extraction"""
        result = {}
        
        # Try to extract key-value pairs using regex
        pairs = re.findall(r'(\w+)\s*[:=]\s*("[^"]*"|\'[^\']*\'|[^,}]+)', answer_str)
        
        for key, value in pairs:
            key = key.strip().lower()
            value = value.strip()
            
            # Clean value
            value = re.sub(r'^["\'\s]+|["\'\s]+$', '', value)
            
            # Try to convert based on likely types in the format hint
            if "int" in format_hint or "integer" in format_hint or "count" in key.lower():
                try:
                    result[key] = int(re.search(r'-?\d+', value).group()) if re.search(r'-?\d+', value) else 0
                except:
                    result[key] = 0
            elif "float" in format_hint or "margin" in key.lower() or "revenue" in key.lower():
                try:
                    match = re.search(r'-?\d+\.?\d*', value)
                    result[key] = float(match.group()) if match else 0.0
                except:
                    result[key] = 0.0
            else:
                result[key] = value
        
        # Fallback to simple regex if no pairs found
        if not result:
            category_match = re.search(r'category[:=]\s*"?([^",}]+)"?', answer_str, re.IGNORECASE)
            quantity_match = re.search(r'quantity[:=]\s*"?(\d+)"?', answer_str, re.IGNORECASE)
            customer_match = re.search(r'customer[:=]\s*"?([^",}]+)"?', answer_str, re.IGNORECASE)
            margin_match = re.search(r'margin[:=]\s*"?([\d\.]+)"?', answer_str, re.IGNORECASE)
            
            if category_match:
                result["category"] = category_match.group(1).strip()
            if quantity_match:
                try:
                    result["quantity"] = int(quantity_match.group(1))
                except:
                    result["quantity"] = 0
            if customer_match:
                result["customer"] = customer_match.group(1).strip()
            if margin_match:
                try:
                    result["margin"] = float(margin_match.group(1))
                except:
                    result["margin"] = 0.0
        
        return result

    def _parse_list_format(self, answer_str: str, format_hint: str) -> List[Any]:
        """Parse list format with flexible item extraction"""
        items = []
        
        # Try JSON parsing first
        try:
            parsed = json.loads(self._fix_json_format(answer_str))
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        
        # Extract items from brackets or using regex
        if "[" in answer_str and "]" in answer_str:
            content = answer_str[answer_str.find("[")+1:answer_str.rfind("]")]
            item_strings = [item.strip() for item in content.split(",") if item.strip()]
        else:
            # Look for bullet points or numbered items
            item_strings = re.findall(r'[-*]?\s*(?:\d+\.\s*)?([^{]+?)(?=\s*[-*]|\s*\d+\.|\s*$)', answer_str)
        
        for item_str in item_strings:
            item_str = item_str.strip()
            if not item_str:
                continue
                
            # Try to parse as object if format hint suggests objects
            if "{" in format_hint:
                try:
                    item = self._parse_object_format(item_str, format_hint)
                    items.append(item)
                except:
                    # Fallback to string
                    items.append(self._clean_string_value(item_str))
            else:
                # Simple value parsing
                if format_hint == "list[int]":
                    try:
                        items.append(int(re.search(r'-?\d+', item_str).group()))
                    except:
                        items.append(0)
                elif format_hint == "list[float]":
                    try:
                        match = re.search(r'-?\d+\.?\d*', item_str)
                        items.append(float(match.group()) if match else 0.0)
                    except:
                        items.append(0.0)
                else:
                    items.append(self._clean_string_value(item_str))
        
        return items

    def _clean_string_value(self, value: str) -> str:
        """Clean string values by removing extra quotes and whitespace"""
        return re.sub(r'^["\'\s]+|["\'\s]+$', '', value)

    def parse_citations(self, citations_str: str) -> List[str]:
        """Convert comma-separated citations string to clean list"""
        if not citations_str:
            return []
        
        # Handle JSON array format
        try:
            if citations_str.strip().startswith("[") and citations_str.strip().endswith("]"):
                parsed = json.loads(citations_str)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if item]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Handle comma-separated format
        citations = []
        for part in re.split(r',|;', citations_str):
            part = part.strip()
            if part:
                # Clean common citation formats
                part = re.sub(r'^[\d\s\.\-]+\s*', '', part)  # Remove numbering prefixes
                part = re.sub(r'^\(|\)$', '', part)  # Remove parentheses
                part = re.sub(r'^["\']|["\']$', '', part)  # Remove quotes
                if part:
                    citations.append(part)
        
        return citations

    def sanitize_sql_query(self, sql_query: str) -> str:
        """Safe SQL sanitizer — ensures triple single-quote wrapped exactly once."""
        if not sql_query:
            return sql_query
 
        sql_query = sql_query.strip()
        # Remove markdown fences
        sql_query = re.sub(r"^```sql", "", sql_query, flags=re.IGNORECASE).strip()
        sql_query = re.sub(r"```$", "", sql_query).strip()
        # Ensure semicolon exactly once
        sql_query = sql_query.rstrip(";").strip() + ";"
 
        # Wrap exactly once
        return sql_query
    
    def route(self, state: AgentState) -> Dict[str, Any]:
        """Use Ollama LLM (phi3.5:3.8b-mini-instruct-q4_K_M) to classify question safely."""
        state.trace.append("Routing question using Ollama LLM...")

        system_prompt = "You are a highly strict classifier."
        user_prompt = f"""
            You are a query router. Classify the user question into one of four labels:

            - sql: requires only structured data from database tables (quantities, revenue, 
            totals, averages, counts, margins, top products/customers, date filtering, etc.)

            - rag: requires only unstructured knowledge from documents such as policies, 
            product rules, definitions, textual descriptions, marketing calendar, 
            or KPI documentation.

            - hybrid: requires BOTH (1) lookup of definitions/policies/docs AND (2) 
            computations from structured database tables .


            Routing Rules:
                1. References to “policy”, “definition”, “KPI docs”, “manual”, “rules”, 
                “marketing calendar”, or any document → RAG signal.
                2. Requests for revenue, quantity sold, top products, margins, AOV, 
                totals, averages, aggregates → SQL signal.
                3. If BOTH RAG and SQL signals appear → hybrid.
                4. If only SQL signals appear → sql.
                5. If only RAG signals appear → rag.
                6. Output ONLY one label: sql, rag, hybrid, or general.

            Respond with only the label.
            Question: "{state.question}"
        """
        
        response = self._call_llm(system_prompt, user_prompt)
        llm_answer = response

        token = re.findall(r"[a-z]+", llm_answer.lower())
        token = token[0] if token else ""
        
        state.classification = token
        state.trace.append(f"Routing decision: {token}")
        print(f"Routing decision: {token}")
        return {"classification": token}

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant document chunks"""
        state.trace.append("Retrieving relevant documents...")

        retrieved_chunks = self.retriever.retrieve(state.question, k=8)
        state.retrieved_chunks = retrieved_chunks

        state.trace.append(f"Retrieved {len(retrieved_chunks)} chunks")
        return {"retrieved_chunks": retrieved_chunks}

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Extract constraints from context"""
        state.trace.append("Extracting constraints...")
        constraints = {}
        
        # Simple constraint extraction from retrieved chunks
        full_context = " ".join(c['content'] for c in state.retrieved_chunks).lower()
        
        if 'summer beverages 1997' in full_context:
            constraints.update({
                "date_range": ("1997-06-01", "1997-06-30"),
                "campaign": "Summer Beverages 1997"
            })
        elif 'winter classics 1997' in full_context:
            constraints.update({
                "date_range": ("1997-12-01", "1997-12-31"),
                "campaign": "Winter Classics 1997"
            })
            
        state.constraints = constraints
        state.trace.append(f"Constraints: {list(constraints.keys())}")
        return {"constraints": constraints}

    def generate_sql(self, state: AgentState) -> Dict[str, Any]:
        """Generate SQL using direct Ollama call"""
        state.trace.append("Generating SQL...")
        
        if state.classification == "rag":
            state.sql_query = ""
            state.trace.append("RAG flow - skipping SQL generation")
            return {"sql_query": ""}
        
        system_prompt = "You are an expert SQL developer who ONLY writes valid SQLite queries."

        user_prompt = f"""
        Generate a valid SQLite query for the question below using ONLY the provided database schema.

        ### Critical Rules:
        1. ⚠️ **Table and column names that contain spaces, hyphens, reserved keywords, or special characters MUST be enclosed in double square brackets**, e.g., [Order Details], [Customer Name].
        2. Use explicit JOIN syntax with ON clauses (no implicit joins).
        3. Always end the query with a semicolon (`;`).
        4. Apply all constraints from the context section below.
        5. Output ONLY the SQL query—no markdown, comments, explanations, or extra text.
        6. The entire query must be on a single line with no newline (`\\n`) or line break characters.
        7. Use date literals in 'YYYY-MM-DD' format.
        8. ❗ **Do not invent table or column names. Use ONLY those listed in the schema below.**
        9. If a table name in the schema is shown with brackets (e.g., [Order Details]), you MUST use the same form in your query.

        ### Database Schema:
        {self.schema_info}

        ### Constraints from Context:
        {json.dumps(state.constraints) if state.constraints else 'None'}

        ### Question:
        {state.question}
        """
        try:
            raw_sql = self._call_llm(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            raw_sql = ""
            
        state.sql_query =f'''{self.sanitize_sql_query(raw_sql)}''' 
        state.trace.append(f"Generated SQL: {state.sql_query}")
       
        return {"sql_query": state.sql_query}

    def execute_sql(self, state: AgentState) -> Dict[str, Any]:
        """Execute SQL with safety checks"""
        state.trace.append("Executing SQL...")
        
        if not state.sql_query or state.classification == "rag":
            state.sql_results = []
            state.sql_columns = []
            state.sql_error = ""
            state.trace.append("Skipping execution (RAG/empty query)")
            return {
                "sql_results": [],
                "sql_columns": [],
                "sql_error": ""
            }
        
        results, columns, error = self.sqlite_tool.execute_query(state.sql_query)
        state.sql_results = results
        state.sql_columns = columns
        state.sql_error = error
        
        print(f"Results: {results}, columns: {columns}")
        if error:
            state.trace.append(f"SQL Error: {error}")
        else:
            state.trace.append(f"Got {len(results)} rows")
            
        return {
            "sql_results": results,
            "sql_columns": columns,
            "sql_error": error
        }

    def synthesize(self, state: AgentState) -> Dict[str, Any]:
        """Generate final answer using direct Ollama call"""
        state.trace.append("Synthesizing answer with Ollama...")
        
        # Prepare context string with chunk IDs for citation tracking
        context_str = "\n\n".join([
            f"[{chunk['chunk_id']}]: {chunk['content']}" 
            for chunk in state.retrieved_chunks
        ])
        
        # Convert SQL results to readable format
        results_str = json.dumps(state.sql_results, indent=2) if state.sql_results else "No results"
        
        system_prompt = """
            You are a helpful analytics assistant that produces accurate, concise answers with proper citations.
            Always obey the required JSON output contract exactly.
            You must also estimate a confidence score between 0.0 and 1.0.
        """
        user_prompt = f"""
            You must generate the final structured answer using the following strict Output Contract:

            Output Contract (per question):
            {{
            "id": "rag_policy_beverages_return_days",
            "final_answer": <matches format_hint>,
            "sql": "<last executed SQL or empty string if RAG-only>",
            "confidence": <float between 0.0 and 1.0>,
            "explanation": "<= 2 sentences>",
            "citations": [
                "table_name (for SQL sources)",
                "chunk_id (for RAG documents)"
            ]
            }}

            ========================
            CONFIDENCE RULE
            ========================
            You must compute a confidence score (0.0 - 1.0) based on:
            - Strength and clarity of evidence from SQL Results and/or Document Context  
            - Whether the answer is directly stated or inferred  
            - Presence of conflicting information  
            - Missing SQL Results or Document Context lowers confidence  
            - Exact numeric queries with strong SQL evidence → high confidence (≥ 0.9)  
            - Ambiguous or multi-step inference → medium confidence (0.5–0.8)
            - Weak evidence or partial context → low confidence (≤ 0.5)

            ========================
            GENERAL RULES
            ========================
            1. For *RAG questions* → Use ONLY the Document Context.
            2. For *SQL questions* → Use ONLY the SQL Results.
            3. For *Hybrid questions* → Combine both sources logically.
            4. Explanation must be at most *two sentences*, describing how you derived the answer.
            5. Citations must be a *list*, containing:
                - SQL table names
                - Document chunk_ids
            6. *final_answer must strictly follow the format_hint*.
            7. *sql* must contain the last executed SQL query.
            - If no SQL query ran, return an empty string "".
            8. Return *ONLY the JSON object*, no extra commentary.

            ========================
            DATA
            ========================

            Question:
            {state.question}

            Classification:
            {state.classification}

            Format Hint:
            {state.format_hint}

            SQL Results:
            {results_str}

            Document Context:
            {context_str}

            SQL Error (if any):
            {state.sql_error}

            ========================
            NOW PRODUCE THE EXACT JSON OUTPUT AS SPECIFIED.
            Citations: [comma-separated list of citations]
        """
        try:
            # LLM call
            response_text = self._call_llm(system_prompt, user_prompt)

            # Attempt JSON parsing directly
            parsed = None
            try:
                parsed = json.loads(response_text)
            except:
                # Fix minor JSON issues
                fixed = self._fix_json_format(response_text)
                parsed = json.loads(fixed)

            # Parse final_answer with format_hint rules
            parsed_answer = self.parse_answer(str(parsed.get("final_answer", "")), state.format_hint)

            # Normalize citations
            parsed_citations = self.parse_citations(str(parsed.get("citations", [])))

            state.final_answer = parsed_answer
            state.explanation = parsed.get("explanation", "")
            state.citations = parsed_citations
            state.confidence = float(parsed.get("confidence", 0.5))

        except Exception as e:
            logger.error(f"Synthesis failure: {str(e)}")
            state.final_answer = "Unable to generate answer"
            state.explanation = "Synthesis failed."
            state.citations = []
            state.confidence = 0.1

        state.trace.append(f"Answer generated (confidence={state.confidence})")

        return {
            "final_answer": state.final_answer,
            "explanation": state.explanation,
            "citations": state.citations,
            "confidence": state.confidence
    }

    def should_repair(self, state: AgentState) -> str:
        """Determine if repair is needed with safety checks"""
        # Skip repairs for RAG flows or when max repairs reached
        if (state.classification == "rag" or 
            not state.sql_query or 
            state.repair_count >= state.max_repairs):
            return "done"
            
        # Trigger repair on SQL errors or empty results
        needs_repair = state.sql_error or (not state.sql_results and state.classification in ["sql", "hybrid"])
        
        if needs_repair:
            state.trace.append(f"Repair needed (attempt {state.repair_count+1}/{state.max_repairs})")
            return "repair"
            
        return "done"

    def repair(self, state: AgentState) -> Dict[str, Any]:
        """Attempt SQL repair with constraint preservation"""
        state.repair_count += 1
        state.trace.append(f"Repair attempt #{state.repair_count}")
        
        # Apply specific fixes based on error type
        if "no such table" in state.sql_error.lower():
            # Fix table name quoting for Order Details
            state.sql_query = re.sub(
                r'(?i)Order\s+Details', 
                '"Order Details"', 
                state.sql_query
            )
        elif "syntax error" in state.sql_error.lower():
            state.sql_query = self.sanitize_sql_query(state.sql_query)
        elif "no such column" in state.sql_error.lower():
            # Common column fixes
            fixes = {
                'orderdate': 'OrderDate',
                'unitprice': 'UnitPrice',
                'quantity': 'Quantity',
                'discount': 'Discount',
                'productname': 'ProductName',
                'categoryname': 'CategoryName',
                'companyname': 'CompanyName'
            }
            for bad, good in fixes.items():
                state.sql_query = re.sub(
                    fr'(?i)\b{bad}\b', 
                    good, 
                    state.sql_query
                )
        
        # Re-execute repaired query
        results, columns, error = self.sqlite_tool.execute_query(state.sql_query)
        state.sql_results = results
        state.sql_columns = columns
        state.sql_error = error
        
        return {
            "sql_query": state.sql_query,
            "sql_results": results,
            "sql_error": error,
            "repair_count": state.repair_count
        }

    def build_graph(self):
        """Build LangGraph workflow with direct Ollama integration"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route", self.route)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("plan", self.plan)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("synthesize", self.synthesize)
        workflow.add_node("repair", self.repair)
        
        # Entry point
        workflow.set_entry_point("route")
        
        # Main flow
        workflow.add_edge("route", "retrieve")
        workflow.add_edge("retrieve", "plan")
        
        # Conditional routing after planning
        workflow.add_conditional_edges(
            "plan",
            lambda s: s.classification,
            {
                "rag": "synthesize",
                "sql": "generate_sql",
                "hybrid": "generate_sql"
            }
        )
        
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "synthesize")
        
        # Repair loop with safety limits
        workflow.add_conditional_edges(
            "synthesize",
            self.should_repair,
            {
                "repair": "repair",
                "done": END
            }
        )
        
        workflow.add_edge("repair", "execute_sql")
        
        return workflow.compile()

