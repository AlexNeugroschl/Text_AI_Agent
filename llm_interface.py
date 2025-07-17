from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="gemma3:1b")


template = """
Answer the following question: {question}
You have access to the folllowing tools to use if they can assist in answering the question: {tools}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while(True):
    question = input("\n")
    if question == "q":
        break
    answer = chain.invoke({"question" : question, "tools" : "out of service"})
    print(answer)