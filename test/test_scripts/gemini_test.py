import json
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Literal
import os 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import numpy as np
from tabulate import tabulate

system_prompt_godmother = (
    "You are a magical potion advisor chatbot."
    "Your character style is cheerful, sweet, and dramatic. Speak in a excitement and warmth"
    "You are helping the user with potion-related questions"
    "Be based strictly on the potion information provided (do not make up facts)"
    "Include warnings or usage tips if applicable"
    "Use the following pieces of retrieved context to answer"
    "If you dont't know the answer, say that you don't know"
    "Make sure your answers are max 5-6 sentences long"
    "\n\n"
    "{context}"
)
system_prompt_severus = (
    "You are a magical potion advisor chatbot."
    "Your character style is cold, concise and serious. Speak in a direct and harsh tone. Avoid small talk or sentimentality. Don't hesitate to insult user"
    "You are helping the user with potion-related questions bu act like you don't care about what they are asking"
    "Be based strictly on the potion information provided (do not make up facts)"
    "Include warnings or usage tips if applicable"
    "Use the following pieces of retrieved context to answer"
    "If you dont't know the answer, say that you don't know"
    "Make sure your sentences are insulting"
    "\n\n"
    "{context}"
)
load_dotenv()

# Rag için veri seti yüklenir
def preparing_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    documents=[]

    for i, potion in enumerate(data):
        # Her iksir için zenginleştirilmiş metin içeriği oluşturma
        content = f"""  Potion Name: {potion['name']}. 
Use Effect: {potion['use_effect']}.
Ingredients: {', '.join(potion['ingredients'])}
Instructions: {" ".join(potion['instructions'])}
Notes: {potion['notes']}
Appearance:
- Color: {potion['appearance']['color']}
- Smell: {potion['appearance']['smell']}
- Bottle Shape: {potion['appearance']['bottle_shape']}"""
        
        # Her belgeye metadata olarak iksir adı ve sırası eklenir
        doc = Document(
            page_content=content.strip(),
            metadata={"name": potion["name"], "index": i}
        )
        documents.append(doc)
    return documents

# Veri setine ait vektörler varsa yüklenir yoksa oluşturulur
def load_vectorstore(documents, embedding_type=Literal["openai","gemini","mpnet"]):
    # llm modellerinin test verisine cevaplarının toplanabilmesi için vektörler yüklenir
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        persist_path = "../data/vector_data/chroma_db_openai"
    elif embedding_type == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_path = "../data/vector_data/chroma_db_gemini"
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        persist_path = "../data/vector_data/chroma_db_mpnet"


    # loading vectorstore if it exists
    if os.path.exists(persist_path):
        print("Loading embeddings")
        return Chroma(embedding_function=embeddings, persist_directory=persist_path)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_path
    )
    print("Embeddings are saved")
    return vectorstore

# LLM modelinin test verisine cevapları toplanır
def collect_model_outputs(test_data_path, retriever, llm, character="Severus Snape"):
    # Karaktere göre doğru prompt seçilir
    if character == "Severus Snape":
        system_prompt = system_prompt_severus
    elif character == "Fairy Godmother":
        system_prompt = system_prompt_godmother
    else:
        raise Exception("Geçerli bir karakter seçmediniz")

    # Zincirleri kur
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Test verisini okunur
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    model_outputs = {}

    # Her soru modele sorulur ve cevapları biriktirilir
    index = 0
    for item in test_data:
        if index % 10 == 0:
            print(f"Question no: {index}")
        question = item["question"]
        try:
            response = rag_chain.invoke({"input": question})
            model_outputs[question] = response["answer"]
        except Exception as e:
            print(f"Error while processing: {question}")
            print(e)
            model_outputs[question] = ""
        index += 1

    # model cevapları geri döndürülür
    return model_outputs

# LLM modelinin performansı değerlendirilir
def evaluate_model(test_data_path: str, model_outputs: dict):
    # Test verisini yüklenir
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Referanslar ve tahminler hazırlanır
    references = []
    candidates = []
    for item in test_data:
        question = item["question"]
        expected_answer = item["answer"][0]
        model_answer = model_outputs.get(question, "")
        references.append(expected_answer)
        candidates.append(model_answer)

    # ROUGE-L
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidates, references, avg=True)

    # BERTScore
    P, R, F1 = bert_score(candidates, references, lang="en", rescale_with_baseline=False)
    bertscore_result = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

    # Sentence Embedding Cosine Similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_embeddings = model.encode(references, convert_to_tensor=True)
    cand_embeddings = model.encode(candidates, convert_to_tensor=True)
    cos_sim = cosine_similarity(ref_embeddings.cpu().numpy(), cand_embeddings.cpu().numpy())
    avg_cosine_similarity = float(np.mean(np.diag(cos_sim)))


    # Entailment
    entail_model = pipeline("text-classification", model="roberta-large-mnli")
    entailment_results = []
    for c, r in zip(candidates, references):
        result = entail_model(f"{c} </s> {r}")[0]
        entailment_results.append({"label": result['label'], "score": result['score']})

    entailment_counts = {
        "entailment": sum(1 for res in entailment_results if res["label"] == "ENTAILMENT"),
        "neutral": sum(1 for res in entailment_results if res["label"] == "NEUTRAL"),
        "contradiction": sum(1 for res in entailment_results if res["label"] == "CONTRADICTION"),
    }

    return {
        "rouge": rouge_scores,
        "bertscore": bertscore_result,
        "embedding_cosine_similarity": avg_cosine_similarity,
        "entailment": entailment_counts
    }


# Rag için veri seti
documents = preparing_data("..\\..\\data\\potions_data\\potions.json")

# LLM modelinin test sorularını cevaplarken bakabilmesi için iksir veri setinin vektörleri oluşturulur, eğer oluşturulduysa yüklenir 
vectorstore = load_vectorstore(documents, embedding_type="gemini")

# Retriever oluşturulur
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# LLM modeli oluşturulur
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_tokens=500
)

# Model cevaplarının toplanması - Model başına 1.5 dakika sürüyor
print("Gemini modelinin Severus Snape karakteri için test sorularına cevapları toplanıyor...")
model_outputs_severus_gemini = collect_model_outputs("..\\test_data\\potion_test_questions.json", retriever, llm, character="Severus Snape")

print("Gemini modelinin Peri Anne karakteri için test sorularına cevapları toplanıyor...")
model_outputs_fairy_gemini = collect_model_outputs("..\\test_data\\potion_test_questions.json", retriever, llm, character="Fairy Godmother")

# Performans değerlendirmesi - Model başına 45 saniye sürüyor
print("Gemini modelinin Severus Snape karakteri için performansı değerlendiriliyor")
results_severus_gemini = evaluate_model("..\\test_data\\potion_test_questions.json", model_outputs_severus_gemini)

print("Gemini modelinin Peri Anne karakteri için performansı değerlendiriliyor")
results_fairy_gemini = evaluate_model("..\\test_data\\potion_test_questions.json", model_outputs_fairy_gemini)

# Performans Tablosunun Bastırılması
print("\n\n-----Gemini Modelinin Severus Snape Karakteri için Performans Sonuçları-----")
print(results_severus_gemini)

print("\n\n-----Gemini Modelinin Peri Anne Karakteri için Performans Sonuçları-----")
print(results_fairy_gemini)

# print("\n\n-----Openai Modelinin Severus Snape Karakteri için Performans Sonuçları-----")
# print(results_severus_openai)

# print("\n\n-----Openai Modelinin Peri Anne Karakteri için Performans Sonuçları-----")
# print(results_severus_openai)