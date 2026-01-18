import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from tools import count_letters, compare_numbers

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOllama(model="llama3.2:3b")


class Chat:
    def __init__(self, llm: str="llama3.2:3b", save_history: bool=True, state: AgentState=None, tools: list = None):
        if state is None:
            state = AgentState(messages=[])
        if tools is None:
            tools = []
        self.llm = ChatOllama(model=llm).bind_tools(tools)
        self.save_history = save_history
        self.state = state
        self.tools = tools
        self.graph = StateGraph(AgentState)

        self.graph.add_node("our_agent", self.model_call)
        tool_node = ToolNode(tools=tools)
        self.graph.add_node("tool_node", tool_node)

    
        self.graph.set_entry_point("our_agent")
        self.graph.add_conditional_edges(
            "our_agent",
            self.should_call_tool,
            {
                True : "tool_node",
                False : END,
            },
        )
        self.graph.add_edge("tool_node", "our_agent")
        self.app = self.graph.compile()



    def model_call(self, state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are a helpful assistant.")
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def should_call_tool(self, state: AgentState) -> bool:
        last_message = state["messages"][-1] if state["messages"] else None
        has_tool_calls = isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and bool(last_message.tool_calls)
        if has_tool_calls:
            print(f"DEBUG: Tools to call: {[tc['name'] for tc in last_message.tool_calls]}")
        print(f"DEBUG: should_call_tool returning: {has_tool_calls}")
        return has_tool_calls
    
    def chat(self, user_input: str) -> str:
        user_message = HumanMessage(content=user_input)
        
        # Create a fresh state for this invocation
        input_state = {"messages": self.state["messages"] + [user_message]}
        print(f"DEBUG: Input state messages count: {len(input_state['messages'])}")

        try:
            print("DEBUG: About to invoke graph...")
            result = self.app.invoke(input_state)
            print(f"DEBUG: Graph invoked successfully. Result type: {type(result)}")
            print(f"DEBUG: Result messages count: {len(result['messages'])}")
            
            self.state = result

            ai_message = self.state["messages"][-1]
            print(f"DEBUG: Last message type: {type(ai_message).__name__}")
            
            if isinstance(ai_message, AIMessage):
                if ai_message.content:
                    print(f"DEBUG: AI Response: {ai_message.content}")
                return ai_message.content
            elif isinstance(ai_message, ToolMessage):
                print(f"DEBUG: Tool result: {ai_message.content}")
                return f"Tool result: {ai_message.content}"
            else:
                last_msg_type = type(ai_message).__name__
                return f"Error: Last message is {last_msg_type}, not AIMessage. Content: {getattr(ai_message, 'content', 'N/A')}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n=== DETAILED ERROR ===\n{error_details}\n===================\n")
            return f"Error during chat invocation: {str(e)}"
        
if __name__ == "__main__":
    chat_instance = Chat(llm="functiongemma:270m", tools=[count_letters, compare_numbers])
    print("Chatbot is ready! Type 'q' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            print("Exiting chat...")
            break
        response = chat_instance.chat(user_input)
        print(f"Bot: {response}")