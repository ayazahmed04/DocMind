from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("paper.pdf")
pages = loader.load()
text = pages[0].page_content

if len(pages) > 1:
    print(f"Loaded {len(pages)} pages. First page content:\n{text[:500]}...")
