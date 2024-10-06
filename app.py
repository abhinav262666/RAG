import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from transformers import AutoTokenizer

# Step 1: Set up Chroma Client and Collection
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ncert_docs"

# Create a Persistent Chroma Client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Set up the embedding function using SentenceTransformer
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

# Create a collection in Chroma
collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# Step 2: Extract text from PDF
with pdfplumber.open("data/iesc111.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

# Step 3: Split text into paragraphs
paragraphs = text.split("\n\n")  # Split by paragraphs

chunk_size = 250  # Set the max chunk size in tokens
overlap_size = int(chunk_size * 0.5)  # Set overlap size as 50% of chunk size
effective_chunk_size = chunk_size - overlap_size  # Effective chunk size for large paragraphs (125 tokens)

chunks = []

# Process each paragraph
for paragraph in paragraphs:
    tokenized_paragraph = tokenizer.tokenize(paragraph)

    # If the paragraph has fewer tokens than chunk_size, add it as a single chunk
    if len(tokenized_paragraph) <= chunk_size:
        chunks.append(tokenizer.convert_tokens_to_string(tokenized_paragraph))
    else:
        # For large paragraphs, create overlapping chunks
        for i in range(0, len(tokenized_paragraph), effective_chunk_size):
            chunk = tokenized_paragraph[i:i + chunk_size]  # Create chunk with overlap
            
            # Add a check to ensure no chunk exceeds the max token length (512 tokens)
            if len(chunk) > 512:
                print(f"Error: Chunk of size {len(chunk)} exceeds the max limit of 512 tokens.")
            
            chunks.append(tokenizer.convert_tokens_to_string(chunk))

# At this point, `chunks` contains all the text split into proper token-based chunks with overlap
# print("These are the chunks------------------------------------------->:", list(chunks))

# Step 4: Add embeddings and chunks to the Chroma collection
try:
    collection.add(
        documents=chunks,
        ids=[f"chunk{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i} for i in range(len(chunks))],
    )
except Exception as e:
    print(f"Error while adding documents to the collection: {e}")

# Step 5: Query the collection
query_text = "Sound is produced by vibrating objects. The matter or substance through which sound is transmitted is called a medium."

# Tokenize and truncate query text if needed
tokenized_query = tokenizer.tokenize(query_text)
if len(tokenized_query) > 512:
    print(f"Warning: Query text is too long ({len(tokenized_query)} tokens). Truncating to 512 tokens.")
    tokenized_query = tokenized_query[:512]

# Perform the query on the collection
try:
    query_results = collection.query(
        query_texts=[tokenizer.convert_tokens_to_string(tokenized_query)],  # Use the truncated query text
        n_results=2,
    )

    # Print query results
    print("Documents:", query_results["documents"])
    print("IDs:", query_results["ids"])
    print("Distances:", query_results["distances"])
    print("Metadatas:", query_results["metadatas"])
except Exception as e:
    print(f"Error during querying: {e}")
