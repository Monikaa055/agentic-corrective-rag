from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
# prompt =hub.pull("rlm/rag-prompt")
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
""")

generation_chain = prompt | llm | parser