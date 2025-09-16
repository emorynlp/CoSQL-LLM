from llm import gptcall

system = "You are a helpful assistant."

def GQR(raw_question: str) -> str:
    prompt = """Below is a conversation between a student and an SQL expert, who is writing queries for the student's requests. Decide whether the student's last question requires context from the previous question. Unless the student is explicitly asking for modifications on what is already gathered by previous expert SQL responses, you can assume it isn't based on the previous requests. Don't make assumptions, don't assume what is "likely". If it is not possible to determine, assume it is not based on previous questions. Begin with \"Modifications required: \" and explain what modifications the student is requesting, then end with \"Answer: \" and then output simply 'yes' or 'no'. Don't include any text afterwards.\n\nHere's the conversation:\n"""
    
    
    count = 0
    conversation = ""
    for u in raw_question.split('\n'):
        if count %2 == 0:
            if "|" in u:
                p = """Rewrite the following question such that it doesn't need the clarification question. Output your answer as "question: <your answer>.\n"""
                p+= u
                response = gptcall(system,p)
                q = response.split("question: ")[-1].strip()
                conversation += f"Student: {q}\n"
            else:
                conversation += f"Student: {u}\n"
        else:
            conversation += f"Expert: {u}\n"
        count += 1
    prompt += conversation
    prompt += "\nModifications required: "
    # print(prompt)

    response = gptcall(system,prompt)
    r = response.split("Answer: ")[-1].strip()
    
    
    if r == 'yes':
        prompt = """Below is a conversation between a student and an SQL expert, who is writing queries for the student's requests. Decide whether the student's last question requires context from the previous question. Unless the student is explicitly asking for modifications on what is already gathered by previous expert SQL responses, you can assume it isn't based on the previous requests. Don't make assumptions, don't assume what is "likely". If it is not possible to determine, assume it is not based on previous questions. Begin with \"Modifications required: \" and explain what modifications the student is requesting. Next, put "Is the last question missing information from previous queries: ", and explain if it's missing information. Lastly, put \"Answer: \" and then output simply 'yes' or 'no'. If there is no missing information, then your final answer will be no. Don't include any text afterwards.\n\nHere's the conversation:\n"""
        prompt += conversation
        prompt += "\nIs the last question missing information: "
        response = gptcall(system,prompt)
        r = response.split("Answer: ")[-1].strip()
        if r == "yes":
            prompt = """Below is a conversation between a student and an SQL expert, who is writing queries for the student's requests. Please rewrite the student's last question so that the SQL expert could respond correctly without needing to rely on information from previous parts of the conversation. Ensure that the revised question contains enough context for the expert to provide the correct answer without referencing any earlier parts of the discussion. Don't include any strategy about how to find the answer, just simply restate the question. Make sure that your summary question asks for the exact same information that the student's last question explicitly asked, with nothing additional. Begin with \"Conversation context required: \" and explain what context you are adding to the user's final question. Then, "Request: ", followed by what information the student is explicitly (stated) requesting only in the last question, and then end with \"Revised question: \" and then repeat the student's explicit request in question form. Don't include any text afterwards.\n\nHere's the conversation:\n"""
            prompt += conversation
            prompt += "\nConversation context required: "
            response = gptcall(system,prompt)
            newq = response.split("Revised question: ")[-1].strip()
            return newq
    # print(conversation)
    newq = conversation.split("Student: ")[-1].strip()
    return newq