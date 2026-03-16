# Wiki-RAG (Retrieval-Augmented Generation)

This is a web application which uses RAG to extract information from pre-ingested wikipedia documents.

## Features
 - Get answers from the already ingested wikipedia documents in QDrant
 - Support of web search using Tavily if related documents are not available
 - Reduced LLM calls with semantic cache using redis
 - Session management for different chat sessions using MySQL and MongoDB
 - Implemented Corrective-RAG by rating context relevance with LLM
 - Implemented long term memory and personalization with LangGraph and Postgres
 - Stream the answers from LangGraph and get realtime processing updates in the UI

## Application Snap Shots
  <p>Login Page</p>
  <img src="https://github.com/user-attachments/assets/bea4ad28-2176-41c7-86e4-3ee5960a1c07" width=50% height=50%>

  <p>Chat Page</p>
  <img src="https://github.com/user-attachments/assets/fbd7ff07-01e5-4f80-90ae-e7c90074034a" width=50% height=50%>

  <p>Chat Streaming</p>
  <video src="https://github.com/user-attachments/assets/ba9a28ae-7560-40e8-9e20-db0b73152e28" width=50% height=50%></video>
	
## Technologies used

Frontend - `Streamlit`

Backend - `Python`, `LangGraph`, `Python`, `redis`, `MongoDB`, `MySQL`, `PostgreSQL`

Deployment - `Docker`, `Kubernetes`, `Github Actions`

## Steps to run locally

### Step 1. Clone the repositories
#### Backend: https://github.com/spandanx/RAGwithCache


### Step 2. Install required softwares

`Miniconda`

### Step 3. Prepare backend

#### Create new environment
<p>Open miniconda console. Run the below commands </p>

```
conda create -n env-name python=3.10
conda activate env-name
```

#### Install required packages
```
python -m pip install -r requirements.txt
```

#### Run the python application
```
streamlit run ui.py
```
The (.env) environment file should contain OPENAI_API_KEY and TAVILY_API_KEY

## Architecture
### Functional Diagram
![FunctionalDiagram_FunctionalDiagram_RAG_CACHE](https://github.com/user-attachments/assets/404a8585-ea4d-4113-a7c5-ea957ce65956)

### LangGraph Node Diagram
![LANGGRAPH_NODE_DIAGRAM](https://github.com/user-attachments/assets/3f992b86-f528-4082-80d0-687c2d02e5b0)

### Login Process
![LOGIN](https://github.com/user-attachments/assets/2de11849-63d7-44a3-90a9-de27949f81da)

### Session Management

![SESSION_MANAGEMENT_UI](https://github.com/user-attachments/assets/ae61d187-5916-4071-8120-4c94b1f97077)

<p>Create New session</p>

![SESSION_MANAGEMENT_CREATE_SESSION](https://github.com/user-attachments/assets/455c53a4-89be-4e7d-8196-0a33c02624a1)

<p>Store chat</p>

![SESSION_MANAGEMENT_STORE_CHAT](https://github.com/user-attachments/assets/67f7bcac-8671-45e8-a813-f8f19ae7888a)

<p>Ragas Score</p>

![RAGAS_SCORE](https://github.com/user-attachments/assets/06a83215-8ef9-4b1b-a1da-515b35a2bb83)

### Youtube link
https://youtu.be/eFo-KTK-K-o
