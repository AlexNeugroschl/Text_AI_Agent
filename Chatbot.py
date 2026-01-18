from AI_Agent import Chat
from tools import count_letters, compare_numbers


chat_instance = Chat(llm="functiongemma:270m", tools=[count_letters, compare_numbers])
print("Chatbot is ready! Type 'q' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'q':
        print("Exiting chat...")
        break
    response = chat_instance.chat(user_input)
    print(f"Bot: {response}")