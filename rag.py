import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------- 1. Load data ----------
import glob

data = []
for file_path in glob.glob("data/*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data.extend(json.load(f))

docs = []
for item in data:
    # Embed act name + section + title INTO the chunk text so retrieval can distinguish sections
    page_content = f"[{item['act']} | Section {item['section']} | {item['title']}]\n{item['text']}"
    docs.append(Document(
        page_content=page_content,
        metadata={
            "act": item["act"],
            "section": item["section"],
            "title": item["title"],
            "category": item["category"],
        },
    ))

print(f"Loaded {len(docs)} chunks")

# ---------- 2. Build retrievers ----------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

faiss_store = FAISS.from_documents(docs, embeddings)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 4})

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

# ---------- 3. Prompt + LLM ----------
llm = OllamaLLM(model="llama3.1:8b", temperature=0.5)

prompt = PromptTemplate.from_template("""You are an Indian legal rights assistant. Answer the user's question using ONLY the retrieved provisions below.

Rules:
- Cite the act name and section number for every claim (e.g., "POSH Act 2013, Section 9").
- If the retrieved provisions do not cover the question, say: "I don't have enough information on this specific issue in my knowledge base. Please consult a legal aid clinic."
- Do NOT invent sections, timelines, or procedures.
- Answer in plain, simple language — like explaining to a friend. Do not copy legal text verbatim.
- If there are steps the user should take, list them clearly.

Retrieved provisions:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# ---------- 4. Chain ----------
chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ---------- 5. Interactive loop ----------
if __name__ == "__main__":
    print("\nKnow Your Rights — All Act assistant. Type 'quit' to exit.\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        # Show what was retrieved (for debugging — remove later)
        retrieved = hybrid_retriever.invoke(q)
        print("\n[Retrieved]:")
        for d in retrieved:
            print(f"  - {d.metadata['act']} | Section {d.metadata['section']} | {d.metadata['title']}")

        print("\n[Answer]:")
        print(chain.invoke(q))
        print("\n" + "=" * 70 + "\n")