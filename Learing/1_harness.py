def fake_llm(messages):
    last = messages[-1]
    if last["role"] == "user":                       # first turn: ask for a tool
        return {"tool_call": {"name": "add", "args": {"a": 2, "b": 3}}}
    return {"text": f"The answer is {last['content']}."}  # after the tool: answer

def add(a, b):
    return a + b

tools = {"add": add}
messages = [{"role": "user", "content": "What is 2 + 3?"}]

while True:                                          # the agent loop
    reply = fake_llm(messages)
    if "tool_call" in reply:
        call = reply["tool_call"]
        result = tools[call["name"]](**call["args"])  # run the tool
        messages.append({"role": "tool", "content": str(result)})
    else:
        print(reply["text"])
        break