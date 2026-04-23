import sys
import os
from pathlib import Path

from dotenv import load_dotenv

if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).resolve().parents[1]
    script_dir = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == script_dir:
        sys.path.pop(0)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

load_dotenv()
os.environ.setdefault("USER_AGENT", "agentic-rag/1.0")

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

def decide_to_generate(state):
    print("--ASSESS GRADED DOCUMENTS--")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB_SEARCH---"
        )
        return WEB_SEARCH
    else:
        print("---DECISION:GENERATE:---")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state:GraphState)->str:
    
    print("---CHECK HALLUCINATIONS")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation":generation}
    )

    if hallucination_grade:= score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({"question":question,"generation":generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else: 
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION")
    question = state["question"]
    source:RouteQuery=question_router.invoke({"question":question})
    
    if source.datasource==WEB_SEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
       print("---ROUTE QUESTION TO RAG---")
       return RETRIEVE
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    path_map={
        WEB_SEARCH:WEB_SEARCH,
        RETRIEVE:RETRIEVE
    }
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")