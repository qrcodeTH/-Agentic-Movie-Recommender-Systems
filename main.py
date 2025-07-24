# main.py
# Defines the agent's graph and runs the application.

from langgraph.graph import StateGraph, END

# Import the state and all node functions
from agent_nodes import (
    AgentState,
    extract_intent_node,
    verify_title_and_plan_node,
    search_by_title_node,
    search_by_category_node,
    analyze_candidates_node,
    format_recommendation_node
)

# --- Define the Graph ---

def route_search(state: AgentState):
    """Router to decide which search path to take."""
    print(f"🗺️ Routing based on: {state['request_type']}")
    return state['request_type']

# Create a new graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.set_entry_point("extract_intent")
workflow.add_node("extract_intent", extract_intent_node)
workflow.add_node("verify_title_and_plan", verify_title_and_plan_node)
workflow.add_node("search_by_title", search_by_title_node)
workflow.add_node("search_by_category", search_by_category_node)
workflow.add_node("analyze_candidates", analyze_candidates_node)
workflow.add_node("format_recommendation", format_recommendation_node)

# Add edges to define the flow
workflow.add_edge("extract_intent", "verify_title_and_plan")
workflow.add_conditional_edges(
    "verify_title_and_plan",
    route_search,
    {
        "TITLE": "search_by_title",
        "CATEGORY": "search_by_category"
    }
)
workflow.add_edge("search_by_title", "analyze_candidates")
workflow.add_edge("search_by_category", "analyze_candidates")
workflow.add_edge("analyze_candidates", "format_recommendation")
workflow.add_edge("format_recommendation", END)

# Compile the graph into a runnable app
app = workflow.compile()


# --- Run the Agent ---
if __name__ == "__main__":
    # --- Edit the user question here ---
    user_question = "Recommend a movie like World War Z for me"

    print("\n" + "="*50)
    print("🚀 Running CineAgent...")
    print(f"💬 User Question: {user_question}")
    print("="*50 + "\n")
    
    initial_state = {"question": user_question}
    final_state_output = None
    
    # Stream the execution of the graph
    for output in app.stream(initial_state, {"recursion_limit": 15}):
        for key, value in output.items():
            print(f"--- Finished Node: {key} ---")
            final_state_output = value
    
    print("\n" + "="*50)
    print("✅ Final Recommendation:")
    print("="*50)
    if final_state_output and 'recommendation' in final_state_output:
        print(final_state_output['recommendation'])
    else:
        print("Sorry, something went wrong and I could not generate a recommendation.")
