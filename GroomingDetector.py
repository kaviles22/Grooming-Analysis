import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
import pandas as pd

PROMPT_TEMPLATE = """
                You are an expert on detecting grooming on chat conversations. 
                This are some grooming chat examples, keep in mind that conversation 
                messages are separated by a | character. 

                {context}

                ---
                Taking into account the previous examples, do you identify any grooming behavior 
                in the next chat? Answer if the conversation is grooming or not, and
                give the literal text that makes you think so.

                {question}
                """
def join_docs(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)

class GroomingDetector:
    def __init__(self, qdrant_url, qdrant_key, temperature=0):
        self.qdrant_url = qdrant_url
        self.qdrant_key = qdrant_key
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        self.qdrant, self.retriever = self.load_vectorstore()
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.chain = self.load_chain()

    def load_vectorstore(self):
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_function,
            collection_name="groom_chats",
            url=self.qdrant_url,
            api_key=self.qdrant_key,
        )
        retriever = qdrant.as_retriever()
        return qdrant, retriever
    
    def load_chain(self):
        chain = (
            {"context": self.retriever | join_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def invoke(self, text):
        return self.chain.invoke(text)