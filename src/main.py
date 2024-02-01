from langchain_community.vectorstores import Chroma
import gradio as gr
from vertexai.language_models import ChatModel
from langchain_google_vertexai import VertexAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


def createSession():
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template
    )
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    persist_directory = "./docData"

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    model = VertexAI(model_name="gemini-pro", temperature=0.3)

    qa = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    return qa


with gr.Blocks() as demo:
    qa = createSession()

    chatbot = gr.Chatbot(
        [
            (
                "Hello! I'm an AI assistance helping you with any questions about langchain.",
                None,
            )
        ],
        label="Chatbot powered by Vertex AI | Google Cloud",
    )
    msg = gr.Textbox(placeholder="Type your response here", label="Response")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        results = qa.invoke({"query": message})
        bot_message = results["result"] + "\n\n" + "Here are some relevant website."
        for result in results["source_documents"]:
            bot_message += "\n" + result.metadata["source"]

        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)
