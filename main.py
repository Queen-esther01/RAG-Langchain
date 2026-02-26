import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# embeddings_model = OpenAIEmbeddings()
# embeddings = embeddings_model.embed_documents(
#     [
#         "This is the Fundamentals of RAG course.",
#         "Educative is an AI-powered online learning platform.",
#         "There are several Generative AI courses available on Educative.",
#         "I am writing this using my keyboard.",
#         "JavaScript is a good programming language"
#     ]
# )

# Ingest Documents
documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
    "At Educative, we think RAG is the future of AI!"
]
db = Chroma.from_texts(documents, OpenAIEmbeddings())

# Retrieve results from semantic search
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)
question = "What is the future of AI?"
result = retriever.invoke(question)
# print(result[0].page_content)

# Define augmented query
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
augmented_query = custom_rag_prompt.format(context=result[0].page_content, question="What is the future of AI?")

# Integrate Chat Model
llm = ChatOpenAI(model="gpt-4o")

print(f"retriever: {retriever}")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
# response = rag_chain.invoke("What is the future of AI?")
# print(f"Response 1: {response}")

#-------OR--------
passthrough_output = RunnablePassthrough().invoke(question)
retriever_output = retriever.invoke(passthrough_output)
custom_prompt_output = custom_rag_prompt.invoke({"context": retriever_output[0].page_content, "question": question})
llm_output = llm.invoke(custom_prompt_output)
final_output = StrOutputParser().invoke(llm_output)
print(f"Response 2: {final_output}")

#for evaluation - ragas & deep eval, datasets at https://huggingface.co/datasets/vibrantlabsai/WikiEval