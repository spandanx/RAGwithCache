
from langchain_openai import ChatOpenAI
import asyncio

import os
from dotenv import load_dotenv

from main import RAGApplication

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

rag_app = RAGApplication()


sample_queries = [
    "How many states are in Northeast India, and what are they?",
    "What is the total area and population of Northeast India?",
    "Name the highest peak in Northeast India and its height.",
    "Which states form the \"Seven Sisters\"?",
    "What are the official languages of Assam?",
    "List the national parks in Assam.",
    "Describe the climate of Northeast India briefly.",
    "What was the significance of the Battles of Imphal and Kohima?",
    "Which religion has the highest percentage in the Northeast region overall?",
    "Name the largest city in Northeast India by population."
]

expected_responses = [
    "Northeast India comprises eight states: Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura (the \"Seven Sisters\"), and Sikkim.",
    "The region covers 262,184 square kilometers (about 8% of India's area) with a population of 45,772,188 (about 4% of India's).",
    "Kangchenjunga in Sikkim, at 8,586 meters (28,169 feet), shared with Nepal.",
    "Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, and Tripura.",
    "Assamese, Bodo, Meitei (Manipuri), and Bengali.",
    "Manas National Park, Kaziranga National Park, Dibru-Saikhowa National Park, Nameri National Park, and Orang National Park.",
    "Predominantly humid subtropical with hot, humid summers, severe monsoons, mild winters, and high rainfall (over 2,000 mm annually in most areas); varies by altitude from tropical to snow microthermal.",
    "Fought in 1944 during WWII, they marked a decisive Allied victory over Japanese forces, halting their advance into India and turning the Burma Campaign.",
    "Hinduism, at 54.02% of the total population per 2011 census data.",
    "Guwahati in Assam, with 968,549 residents (2011 census)."
]

dataset = []

####### ----------- Sample run
# query = "What was the cricket score in india vs Namibia?"
# response = asyncio.run(rag_app.answer_question(question = query, chat_history = ''))
# print(response)
# x = 1
####### -----------

for query, reference in zip(sample_queries, expected_responses):
    response = asyncio.run(rag_app.answer_question(question=query, chat_history=''))

    relevant_docs = [res["page_content"] for res in response["relevant_documents"]]
    response = response["final_response"]
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


evaluator_llm = LangchainLLMWrapper(llm)
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, AnswerRelevancy, AnswerAccuracy

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), AnswerRelevancy(), AnswerAccuracy()],llm=evaluator_llm)
print(result)
x = 1
#{'context_recall': 0.8400, 'faithfulness': 0.9167, 'factual_correctness(mode=f1)': 0.4470, 'answer_relevancy': 0.7714, 'nv_accuracy': 0.8750}

############### Display in chart
import matplotlib.pyplot as plt

data = result

metrics = list(data.keys())
scores = list(data.values())

plt.figure(figsize=(8, 5))
plt.bar(metrics, scores, color='skyblue')

plt.xlabel('metrics')
plt.ylabel('score')
plt.xticks(rotation=45, ha='right')
plt.title('Ragas evaluation')

plt.show()
############### Display in chart