from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph
from langgraph.graph import END

# =========================
# Load Environment
# =========================

load_dotenv()

# =========================
# OpenAI Model
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================
# Vector Database
# =========================

db = Chroma(
    persist_directory="./db",
    embedding_function=OpenAIEmbeddings()
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# =========================
# State Definition
# =========================

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    route: str   # ← חדש

# =========================
# Tool (Calculator)
# =========================

def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"

# =========================
# Node 1 - Router
# =========================

def router_node(state: AgentState):

    print("\n===================")
    print("ROUTER NODE")
    print("===================")

    question = state["question"]

    # החלטה: Tool או RAG
    if any(op in question for op in ["+", "-", "*", "/"]):
        state["route"] = "tool"
    else:
        state["route"] = "rag"

    print(f"Question: {question}")
    print(f"Route: {state['route']}")

    return state

# =========================
# Router Decision Function
# =========================

def route_decision(state: AgentState):
    return state["route"]

# =========================
# Node 2 - RAG
# =========================

def rag_node(state: AgentState):

    print("\n===================")
    print("RAG NODE")
    print("===================")

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    state["context"] = context

    print("Retrieved Context (preview):")
    print(context[:300])

    return state

# =========================
# Node 3 - Tool
# =========================

def tool_node(state: AgentState):

    print("\n===================")
    print("TOOL NODE")
    print("===================")

    question = state["question"]

    result = calculator_tool(question)

    state["answer"] = result

    print(f"Tool Result: {result}")

    return state

# =========================
# Node 4 - Response
# =========================

def response_node(state: AgentState):

    print("\n===================")
    print("RESPONSE NODE")
    print("===================")

    # אם הגיע מ־Tool
    if state["answer"]:
        return state

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Question:
{state['question']}

Context:
{state['context']}

If the answer is not in the context,
say: "The information was not found."
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state

# =========================
# Build Graph
# =========================

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("tool", tool_node)
graph.add_node("response", response_node)

graph.set_entry_point("router")

# 👇 Router חכם
graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "rag",
        "tool": "tool"
    }
)

# המשך הזרימה
graph.add_edge("rag", "response")
graph.add_edge("tool", "response")
graph.add_edge("response", END)

app = graph.compile()

# =========================
# Main Loop
# =========================

print("\nAgent Ready 🚀")
print("Type 'exit' to quit.\n")

while True:

    question = input("Question: ")

    if question.lower() == "exit":
        break

    result = app.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "route": ""
        }
    )

    print("\nAnswer:")
    print(result["answer"])
    print()