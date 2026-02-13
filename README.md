# Wiki-RAG (Retrieval-Augmented Generation)

This is a web application which uses RAG to extract information from pre-ingested wikipedia documents.

## Features
 - Get answers from the already ingested wikipedia documents
 - Support of web search if documents are not available
 - Reduced LLM calls with semantic cache using redis

## Application Snap Shots
  <p>LoginPage</p>
  

	
## Technologies used

Frontend - `Streamlit`

Backend - `Python`, `LangGraph`, `Python`, `redis`, `MongoDB`, `MySQL`

## Steps to run locally

### Step 1. Clone the repositories
#### Backend: [[https://github.com/spandanx/youtube-comment-analysis-app-python](https://github.com/spandanx/ResumeATS)](https://github.com/spandanx/RAGwithCache)


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

```
#### Run the python application
```
streamlit run ui.py
```
The (.env) environment file should contain OPENAI_API_KEY and TAVILY_API_KEY

## Architecture
### High Level Design Diagram


### Activity Diagram


### Functional Diagram



## Youtube link


