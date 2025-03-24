#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

import os
import json


os.environ["OPENAI_API_KEY"] = ""


# Load dos modelos (Embeddings e LLMs)

embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200)

def loadData():
    pdf_link = "lol.pdf"
    loader = PyPDFLoader(pdf_link, extract_images=False)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 20,
        length_function = len,
        add_start_index = True
    )

    chunks = text_splitter.split_documents(pages)
    vectorDB = Chroma.from_documents(chunks, embedding=embeddings_model)
    retriever = vectorDB.as_retriever(search_kwargs={"k": 3})
    return retriever

def getRelevantDocuments(question):
    retriever = loadData()
    context = retriever.invoke(question)
    return context


def ask(question, llm):
    TEMPLATE = """"
        Você é um especialista em league of legends. Responda a pergunta abaixo utilizando o contexto informado.
        Pergunta: {question}
        Contexto: {context}
    """

    prompt = PromptTemplate(input_variables= ['context', 'question'], template=TEMPLATE)

    sequence = RunnableSequence(prompt | llm)
    context = getRelevantDocuments(question)

    response = sequence.invoke({'context': context, "question": question})

    return response

def lambda_handler(event, context):
    query = event.get('question')
    response = ask(query, llm).content
    return{
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "message": "Tarefa concluída",
            "details": response
        })
    }