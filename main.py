from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
from transformers import AutoTokenizer
# Initialize FastAPI app
import pickle
import requests
app = FastAPI()

# Load pre-trained SVM model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize ChromaDB
CHROMA_DATA_PATH = "chroma_data/"
COLLECTION_NAME = "ncert_docs"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(COLLECTION_NAME)







# API key for WeatherAPI
WEATHER_API_KEY = "120164fc9c4c41818c340544240710"
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Initialize Ollama model
ollama_llm = Ollama(model="llama3")

# Request body model for weather queries
class QueryRequest(BaseModel):
    query: str

# Step 1: Use Ollama to extract the location from the user's query
def extract_location_using_ollama(user_query):
    # Use Ollama to process the user query and extract the location
    context = f"Extract the location, country, or any place mentioned in this query: '{user_query}'. Just return the place."
    # context += "/n" + "If the query is not about knowing the weather of some place and is mentioing something different such as how are you, or hello or tell me a joke, you can use your knowledge and answer those questions"
    
    # Generate a response using the Ollama LLM
    print("Context going to Ollama:", context)
    response = ollama_llm.invoke(context)
    
    # Extract the response text (assumes Ollama will return a simple location or country)
    extracted_location = response.strip()
    
    if not extracted_location:
        raise HTTPException(status_code=400, detail="Could not extract location from the query.")
    
    return extracted_location

# Step 2: Get weather data from WeatherAPI
def get_weather_by_location(location):
    # Call the WeatherAPI to get current weather data
    try:
        response = requests.get(
            WEATHER_API_URL,
            params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        weather_data = response.json()

        # Return weather information
        return {
            "location": weather_data["location"]["name"],
            "region": weather_data["location"]["region"],
            "country": weather_data["location"]["country"],
            "temperature_c": weather_data["current"]["temp_c"],
            "condition": weather_data["current"]["condition"]["text"],
            "wind_kph": weather_data["current"]["wind_kph"],
            "humidity": weather_data["current"]["humidity"],
            "feelslike_c": weather_data["current"]["feelslike_c"]
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

# Step 3: Define API route for weather handling

def format_weather_response(weather_data):
    # Create a readable response with the weather data
    response = (
        f"The current weather in {weather_data['location']}, {weather_data['region']}, {weather_data['country']}:\n"
        f"- Temperature: {weather_data['temperature_c']}°C\n"
        f"- Feels like: {weather_data['feelslike_c']}°C\n"
        f"- Condition: {weather_data['condition']}\n"
        f"- Wind Speed: {weather_data['wind_kph']} km/h\n"
        f"- Humidity: {weather_data['humidity']}%"
    )
    return response

@app.post("/generate-weather/")
async def handle_weather_query(query_request: QueryRequest):
    user_query = query_request.query

    # Use Ollama to extract location
    location = extract_location_using_ollama(user_query)

    # Get weather data for the extracted location
    weather_data = get_weather_by_location(location)

    # filtered_response = filter_weather_response(user_query, weather_data)

    # Format the filtered weather data into a human-readable summary
    formatted_response = format_weather_response(weather_data)

    return {"response": formatted_response}








# Initialize sentence-transformer tokenizer for query embedding
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

# Initialize Ollama model
ollama_llm = Ollama(model="llama3")

# Request body model
class QueryRequest(BaseModel):
    query: str

# Step 1: Search ChromaDB
def search_chroma_db(query, top_n=6):
    # Tokenize and check the length of the query
    # query_embedding = tokenizer.embed([query])[0]
    tokenized_query = tokenizer.tokenize(query)
    # Search the ChromaDB collection with the user query
    # results = collection.query(
    #     query_embeddings=[query_embedding],
    #     n_results=top_n
    # )
    results = collection.query(
        query_texts=[tokenizer.convert_tokens_to_string(tokenized_query)],  # Use the truncated query text
        n_results=2,
    )
    
    relevant_chunks = [result for result in results['documents'][0]]
    
    return relevant_chunks

# Step 2: Generate Response using Ollama
def generate_response(user_query, relevant_chunks=None):
    # If relevant_chunks is provided, use them; otherwise, just use the query
    if relevant_chunks:
        # Construct context with relevant PDF chunks
        context = "You are a helpful assistant. The user has asked a question related to the following document. Use the document to help answer the query.\n\nDocument context:\n" + "\n".join(relevant_chunks) + "\n\nUser query: " + user_query + "\n" + "if you dont find any relevant information from the context attached you can tell the answer from your knowledge but include the sentence in the response that there is no relevant details about this in the given pdf"
    else:
        # Simple context for queries that do not need the PDF
        context = "You are a helpful assistant. Please answer the user's query directly: " + user_query
    
    # Generate a response using the Ollama LLM
    print("Context going to Ollama:", context)
    response = ollama_llm.invoke(context)
    
    return response


def needs_pdf(user_query):
    query_tfidf = vectorizer.transform([user_query])
    return svm_model.predict(query_tfidf)[0] == 1


# Step 3: Define API route for handling user queries
@app.post("/generate-response/")
async def handle_query(query_request: QueryRequest):
    user_query = query_request.query

    if not needs_pdf(user_query):
        # If the query doesn't need the PDF, generate a response directly using Ollama
        response = generate_response(user_query)
        return {"response": response}
    

    # Search ChromaDB for relevant chunks
    relevant_chunks = search_chroma_db(user_query)

    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    # Generate a response using Ollama based on the top chunks
    response = generate_response(user_query, relevant_chunks)

    return {"response": response}

