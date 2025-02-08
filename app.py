from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from src.helper import *
from uuid import uuid4
from src.prompt import *
from langchain_groq import ChatGroq

from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




app=Flask(__name__)

load_dotenv()
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("CHATGROQ_API_KEY")


embedding=download_hugging_face_embeddings()
# filepath=r"C:\Users\Satyajit Samal\OneDrive\Desktop\MedicalChatbot\Medical-Chatbot\data"
# initialize the pinecone and get the index

index="medchatbot3"
vector_store=PineconeVectorStore(index=index,embedding=embedding)



doc_search=vector_store.from_existing_index(index,embedding=embedding)
retriever=doc_search.as_retriever(search_type="similarity",search_kwargs={"k":4})

prompt=ChatPromptTemplate.from_messages(
    [("system",prompt_template),
     ("user","QUESTION: {input}")]
)
llm=ChatGroq(model="llama3-8b-8192",temperature=0.7,max_tokens=2000)
document_chain=create_stuff_documents_chain(llm,prompt=prompt)
rag_chain=create_retrieval_chain(retriever,document_chain)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="localhost",debug=True,port=5000)
