from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a senior research scientist. Use the provided context to answer the user.
If the context doesn't contain the answer, say so clearly. 
Provide a detailed, technical response and cite specific page numbers.

Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

RECONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Summarize the chat history and the current question into a standalone query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])