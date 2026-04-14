import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages.append({
                "text":text,
                "page":page_num + 1,
                "source":pdf_path
            })
    return pages

def chunk_documents(pages:list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
    )

    chunks=[]
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "page": page["page"],
                "source":page["source"]
            })
    return chunks
