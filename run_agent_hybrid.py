# run_agent_hybrid.py
import json
import click
from agent.graph_hybrid import RetailAnalyticsCopilot, AgentState
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
import os

console = Console()

@click.command()
@click.option('--batch', type=click.Path(exists=True, dir_okay=False, file_okay=True), help="JSONL file path")
@click.option('--out', type=click.Path(dir_okay=False, file_okay=True), help="Output file path")

def main(batch, out):
    """Run the Retail Analytics Copilot"""
    console.print(Panel("Retail Analytics Copilot", expand=False))

    # Initialize the copilot
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    DB_PATH = os.path.join(PROJECT_ROOT, "data", "northwind.sqlite")
    DOCS_PATH = os.path.join(PROJECT_ROOT, "docs")

    copilot = RetailAnalyticsCopilot(
        db_path=DB_PATH,
        docs_dir=DOCS_PATH
    )
    # Build the graph
    graph = copilot.build_graph()
    # Read questions from batch file
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
            
    # Process each question
    outputs = []
    for q_data in questions:
        console.print(f"\nProcessing question ID: {q_data['id']}")
        
        # Prepare initial state
        initial_state = AgentState(
            question=q_data['question'],
            format_hint=q_data['format_hint'],
            classification=""
        )
        
        # Run the graph with a higher recursion limit to prevent infinite loops
        try:
            final_state = graph.invoke(initial_state, config={"recursion_limit": 50})
        except Exception as e:
            console.print(f"[red]Error processing question {q_data['id']}: {e}[/red]")
            # Create a default output in case of error
            output = {
                "id": q_data['id'],
                "final_answer": f"Error: {str(e)}",
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing failed with error: {str(e)}",
                "citations": []
            }
            outputs.append(output)
            continue # Skip to next question
        
        # Handle the case where final_state might be a dict
        if isinstance(final_state, dict):
            # Extract values from the dictionary
            output = {
                "id": q_data['id'],
                "final_answer": final_state.get('final_answer', None),
                "sql": final_state.get('sql_query', ''),
                "confidence": final_state.get('confidence', 0.0),
                "explanation": final_state.get('explanation', ''),
                "citations": final_state.get('citations', [])
            }
        else:
            # If it's still an AgentState object, access attributes directly
            output = {
                "id": q_data['id'],
                "final_answer": final_state.final_answer,
                "sql": final_state.sql_query,
                "confidence": final_state.confidence,
                "explanation": final_state.explanation,
                "citations": final_state.citations
            }
        
        outputs.append(output)
        # Convert dict to JSON string for rich JSON display
        console.print(JSON.from_data(output))
    
    # Write outputs to file
    with open(out, 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
    
    console.print(f"\n[green]Outputs written to {out}[/green]")

if __name__ == "__main__":
    main()