import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

load_dotenv()

def create_manim_db(source_dir: str = "./data/videos", output_dir: str = "./manim_rag", year_filter: int = 2019):

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Extracting python files....")
    all_code = []
    metadata = []

    for year_dir in sorted(Path(source_dir).glob("_2*")):
        try:
            year = int(year_dir.name[1:5])
            if year < year_filter:
                continue
        except Exception as e:
            print(f"Trouble accessing the year dirs: {e}")
            continue


        print(f"processing {year_dir.name}")
        for py_file in year_dir.rglob("*.py"):
            try:
                if not py_file.is_file():
                    continue

                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if len(content.strip()) < 50:
                    continue

                header = f"\n{"="*30}\nFILE: {py_file.relative_to(source_dir)}\n{"="*30}\n"
                all_code.append(header+content)
                metadata.append({
                    "file": str(py_file.relative_to(source_dir)),
                    "year": year,
                })

            except Exception as e:
                print(f"Error reading the file {py_file}: {e}")

    print(f"\nExtracted {len(metadata)} files...")

    combined_file = output_path / "all_manim_code.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        for block in all_code:
            f.write(block + "\n\n")
        print(f"Saved all codes to {str(combined_file)}")

    print(f"\nCreating Chroma DB with OpenAI embeddings...")
    documents = []
    for code, meta in zip(all_code, metadata):
        doc = Document(
            page_content=code,
            metadata=meta
        )
        documents.append(doc)

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )

    all_splits = python_splitter.split_documents(documents)

    print(f"Split docs into {len(all_splits)} sub docs")

    print("\nCreating Chroma db now")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chroma_dir = "./chroma_manim_db"

    vector_store = Chroma(
        collection_name="manim_code",
        embedding_function=embeddings,
        persist_directory=chroma_dir
    )

    batch_size = 100
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(all_splits)+batch_size-1)//batch_size}....")
        vector_store.add_documents(batch)

    print("\n\nDone addding docs")
    print(f"----Files: {len(documents)}\n----Chunks: {len(all_splits)}\n----Location: {chroma_dir}\n")

    return vector_store
    
if __name__ == "__main__":

    manim_repo_path = "./data/videos"
    
    if os.path.exists(manim_repo_path):
        create_manim_db(
            source_dir=manim_repo_path,
            output_dir="./data/textfile",
            year_filter=2019
        )
    else:
        print(f"Error: Path not found: {manim_repo_path}")


    