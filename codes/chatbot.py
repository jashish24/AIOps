import ollama

history = []

print("Chatbot ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "content": user_input})

    response = ollama.chat(model="llama3.2", messages=history)
    reply = response.message.content

    history.append({"role": "assistant", "content": reply})
    print(f"\nBot: {reply}\n")
