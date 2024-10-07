# RAG
1. Use the ollama setup file to install ollama locally it will be used for inference.
2. Clone the repo.
3. Create a coda environment and run activate it.
4. Run ```pip install -r requiremnents.txt``` .
5. Run the fastapi app ```uvicorn main:app --reload``` .
6. Run the frontend ```streamlit run frontend.py``` .


SARVAM API for text to speech has been integrated with the frontend so that the user can listen to answers, the second agent tool used is to tell weather, 
in this tool as it if specific we cannot ask any other question. The second tool/endpoint acts as copilot for NCERT chatper 11 : Sound, here you can
ask different queries from the textbook and learn efficienttly, if a curious question has been asked and the agent doesnt find the answer in the textbook,
it will provide relevant result from its knowledge. The second tool is also a normal LLM agent which you can interact with and ask question like "what shoul I
eat today?", "best hindi movies to watch today?"
