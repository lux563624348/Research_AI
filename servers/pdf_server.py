from dotenv import load_dotenv
load_dotenv()# Load environment variables

PDF_FOLDER = "servers/pdfs" 
FAISS_INDEX_PATH = PDF_FOLDER+"/pdf_index"
# pdf_faiss_server.py
import os
import asyncio
from fastmcp import FastMCP
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

SCAN_INTERVAL = 30  # seconds

embeddings = OpenAIEmbeddings()
mcp = FastMCP("PDF_FAISS_Server")

loaded_store = None
known_pdfs = set()

async def load_pdfs(files):
    """Load and return documents from given PDF file list."""
    pages = []
    for pdf in files:
        path = os.path.join(PDF_FOLDER, pdf)
        print(f"📄 Loading {pdf}")
        loader = PyPDFLoader(path)
        async for page in loader.alazy_load():
            pages.append(page)
    return pages

async def build_index_from_files(pdf_files):
    """Build FAISS index from given list of PDFs."""
    global loaded_store
    if not pdf_files:
        print("⚠ No PDFs found, FAISS index cleared")
        loaded_store = None
        return None
    pages = await load_pdfs(pdf_files)
    loaded_store = FAISS.from_documents(pages, embeddings)
    loaded_store.save_local(FAISS_INDEX_PATH)
    print(f"💾 FAISS index rebuilt with {len(pdf_files)} PDFs")
    return loaded_store

async def build_initial_index():
    """Initial full build from all PDFs in folder."""
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    return await build_index_from_files(pdf_files)

async def update_index_if_needed():
    """
    Background task that handles incremental additions and selective rebuilds
    on PDF additions/removals in the PDF_FOLDER.
    """
    global loaded_store, known_pdfs

    while True:
        current_pdfs = {f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")}
        new_pdfs = current_pdfs - known_pdfs
        removed_pdfs = known_pdfs - current_pdfs

        # Incremental addition
        if new_pdfs and loaded_store:
            print(f"📥 New PDFs detected: {', '.join(new_pdfs)}")
            pages = await load_pdfs(new_pdfs)
            loaded_store.add_documents(pages)
            loaded_store.save_local(FAISS_INDEX_PATH)
            print(f"✅ Added {len(new_pdfs)} PDFs to FAISS index")

        # If index missing and new PDFs appear
        elif new_pdfs and not loaded_store:
            print("🛠 Building FAISS index from new PDFs...")
            await build_index_from_files(sorted(current_pdfs))

        # Rebuild index if any PDFs were removed
        if removed_pdfs:
            print(f"🗑 PDFs removed: {', '.join(removed_pdfs)}")
            print(f"🔄 Rebuilding FAISS index from remaining {len(current_pdfs)} PDFs")
            await build_index_from_files(sorted(current_pdfs))

        known_pdfs = current_pdfs
        await asyncio.sleep(SCAN_INTERVAL)

@mcp.tool()
async def update_index() -> str:
    """
    Scan PDF_FOLDER for new or removed PDFs and update the FAISS index accordingly.
    Returns:
        Status message as a string.
    """
    global loaded_store, known_pdfs
    if loaded_store is None:
        await build_initial_index()
        return "FAISS index was missing and has now been built."
    current_pdfs = {f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")}
    new_pdfs = current_pdfs - known_pdfs
    removed_pdfs = known_pdfs - current_pdfs

    messages = []

    if new_pdfs and loaded_store:
        messages.append(f"📥 New PDFs detected: {', '.join(new_pdfs)}")
        pages = await load_pdfs(new_pdfs)
        loaded_store.add_documents(pages)
        loaded_store.save_local(FAISS_INDEX_PATH)
        messages.append(f"✅ Added {len(new_pdfs)} PDFs to FAISS index")

    elif new_pdfs and not loaded_store:
        messages.append("🛠 Building FAISS index from new PDFs...")
        await build_index_from_files(sorted(current_pdfs))
        messages.append("✅ FAISS index built")

    if removed_pdfs:
        messages.append(f"🗑 PDFs removed: {', '.join(removed_pdfs)}")
        messages.append(f"🔄 Rebuilding FAISS index from remaining {len(current_pdfs)} PDFs")
        await build_index_from_files(sorted(current_pdfs))
        messages.append("✅ FAISS index rebuilt after removal")

    known_pdfs = current_pdfs

    if not messages:
        return "No changes detected. FAISS index is up to date."
    return "\n".join(messages)

@mcp.tool()
async def search_pdfs(query: str, k: int = 3) -> list[dict]:
    """
    Search internal locally stored data such as: PDFs for query.
    Args:
        query: search query string
        k: number of results to return
    Returns:
        list of dicts with page, source PDF, and excerpt
    """
    global loaded_store
    if not loaded_store:
        return [{"error": "FAISS index not ready. Add PDFs to the folder and wait for indexing."}]

    results = loaded_store.similarity_search(query, k=k)
    return [
        {
            "pdf": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", -1),
            "excerpt": doc.page_content[:500]
        }
        for doc in results
    ]

async def main():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    global loaded_store, known_pdfs

    if not os.path.exists(FAISS_INDEX_PATH):
        print("🛠 No FAISS index found. Building initial index if PDFs exist...")
        loaded_store = await build_initial_index()
    else:
        print("✅ Loading existing FAISS index...")
        loaded_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    known_pdfs = {f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")}

    # Start background watcher
    asyncio.create_task(update_index_if_needed())    


if __name__ == "__main__":
    #asyncio.run(main())
    # Run the MCP server - note this is blocking and runs its own event loop
    mcp.run(transport="stdio")