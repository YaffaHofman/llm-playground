from openai import OpenAI
from rag import retrieve_relevant_chunks

question = input("What is your question? ")

client = OpenAI()

system_prompt = """
    החזר תשובה לשאלת המשתמש.
    עליך להחזיר נתוני אמת מתוך הקבצים המצורפים
    במידה ולא קבילת מספיק מידע מהקבצים המצורפים,
    עליך להחזיר הודעה על כך שחסר מידע ואין אפשרות לענות על השאלה.
"""


# 1 - Retrieval
documents = retrieve_relevant_chunks(question, 2)

print("Retrieved documents:", documents)
documents_prompt = f"""
    הקבצים המצורפים הם:
    
{"\n\n".join(doc["text"] for doc in documents)}
"""


print("Documents prompt:", documents_prompt)

# 2 - Answer Generation
response = client.responses.create(
    model="gpt-5.4-mini",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": documents_prompt},
        {"role": "user", "content": question}
    ]
)

print("Answer:", response.output_text)