import chromadb

# Step 1: Set up Chroma Client and load the existing collection
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "ncert_docs"

# Load the Persistent Chroma Client (assuming the data is already stored)
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Retrieve the existing collection by name
collection = client.get_collection(COLLECTION_NAME)

# Step 2: Perform a query on the collection
query_results = collection.query(
    query_texts=["Sound is produced by vibrating objects. The matter or substance through which sound is transmitted is called a medium."],
    n_results=2,  # Adjust the number of results you want to retrieve
)
# Count the number of documents (chunks) in the collection
num_chunks = collection.count()
print(f"Number of chunks in the collection: {num_chunks}")

# Step 3: Print the query results
print("Documents:", query_results["documents"])
print("IDs:", query_results["ids"])
print("Distances:", query_results["distances"])
print("Metadatas:", query_results["metadatas"])
