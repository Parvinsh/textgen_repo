from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.schema import Document

def initialize_retrieval_chain(links, llm):
    """Set up the retrieval chain from links."""
    documents = []
    for url in links:
        text = fetch_text_from_url(url)
        if text:
            documents.append(Document(page_content=text))

    if not documents:
        raise ValueError("No valid content found in the provided links.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    docs = text_splitter.split_documents(documents=documents)

    # Create vectorstore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create retrieval chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

def generate_long_answer(retrieval_chain, query, max_length=3000):
    """Generate a long answer by chunking the query."""
    # Split query into subtopics
    subtopics = retrieval_chain.chain.llm.invoke(
        {"input": f"Split the query into subtopics: {query}"}
    )["answer"].split("\n")

    # Generate answers for each subtopic
    long_answer = ""
    for subtopic in subtopics:
        partial_result = retrieval_chain.invoke({"input": subtopic.strip()})
        long_answer += f"\n\n### {subtopic.strip()}\n{partial_result['answer']}"
        if len(long_answer.split()) > max_length:
            break
    return long_answer

