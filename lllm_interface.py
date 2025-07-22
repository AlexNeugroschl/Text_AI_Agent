from langchain_ollama.llms import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.render import render_text_description
from tools import count_letters, compare_numbers

def main():
    try:
        llm = OllamaLLM(model="llama3.2:3b")
    except Exception as e:
        print(f"Error initializing the LLM. Make sure Ollama is running and the model is installed. Error: {e}")
        return

    # --- 2. Define the Tools ---
    tools = [count_letters, compare_numbers]

    # --- 3. Create the Agent Prompt ---
    # This prompt template is specifically designed for ReAct agents.
    # It tells the agent how to think, what tools are available, and how to format its response.
    prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: a JSON dictionary with the arguments for the action. For example: {{"num1": 1.0, "num2": 2.0}}
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """#{{"tool": "count_letters", "tool_input": {{"num1": 1.0, "num2": 2.0}}}}
    prompt = ChatPromptTemplate.from_template(prompt_template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # --- 4. Create the Agent ---
    # `create_react_agent` creates the core logic of the agent.
    # It takes the LLM and the prompt and creates a "runnable" that can process input.
    agent = create_react_agent(llm, tools, prompt)

    # --- 5. Create the Agent Executor ---
    # The `AgentExecutor` wraps the agent and the tools.
    # It's responsible for calling the agent, executing the tools,
    # and passing the results back to the agent until it finds a final answer.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True to see the agent's thought process
        handle_parsing_errors=True # Helps with robustness
    )

    # --- 6. Run the Agent Loop ---
    print("Agent is ready! Ask me something. (Type 'q' to exit)")
    while True:
        try:
            question = input(">> ")
            if question.lower() == "q":
                print("Exiting...")
                break
            if question:
                # Use .invoke() for the new agent executor
                response = agent_executor.invoke({"input": question})
                print("\nFinal Answer:")
                print(response["output"])
                print("-" * 20)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
