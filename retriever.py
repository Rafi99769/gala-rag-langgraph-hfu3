import datasets
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relations: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest['name']}
    )
    for guest in guest_dataset
]

# Create embeddings using sentence-transformers
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store from documents
vector_store = FAISS.from_documents(docs, embeddings)

# Create a retriever from the vector store
st_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation.
    """
    results = st_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "No matching guests found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    description="Retrieves detailed information about gala guests based on their name or relation.",
    func=extract_text
)
