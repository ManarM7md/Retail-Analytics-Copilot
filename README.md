# Retail Analytics Copilot

This project implements a local AI agent that answers retail analytics questions by combining RAG over local documents and SQL queries over a local SQLite database (Northwind).

## Graph Design

1. **Router**: Classifies questions as RAG, SQL, or Hybrid based on keyword analysis
2. **Retriever**: Uses TF-IDF to retrieve relevant document chunks
3. **Planner**: Extracts constraints like date ranges and KPI formulas from documents
4. **NL→SQL Generator**: Creates SQL queries based on question and constraints
5. **Executor**: Runs SQL against the database
6. **Synthesizer**: Formats answers according to the required format hint
7. **Repair Loop**: Attempts to fix SQL errors or invalid outputs up to 2 times

## Trade-offs and Assumptions

1. **CostOfGoods**: Approximated as 70% of UnitPrice when not available in the database
2. **SQL Generation**: Used rule-based approach instead of pure LLM generation for better reliability
3. **Date Handling**: Simplified date filtering based on campaign names in the question
4. **Confidence Scoring**: Basic heuristic combining SQL success and repair attempts
