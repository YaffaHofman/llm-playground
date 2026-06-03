import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import csv
from pathlib import Path
from openai import OpenAI
from rag import retrieve_relevant_chunks

client = OpenAI()

DATASET_PATH = "evaluations/dataset.json"
OUTPUT_PATH = "evaluations/results.csv"

AGENT_MODEL = "gpt-4o-mini"

JUDGE_MODEL = "gpt-4o"

AGENT_SYSTEM_PROMPT = """
    החזר תשובה לשאלת המשתמש.
    עליך להחזיר נתוני אמת מתוך הקבצים המצורפים.
    במידה ולא קיבלת מספיק מידע מהקבצים המצורפים,
    עליך להחזיר הודעה על כך שחסר מידע ואין אפשרות לענות על השאלה.
"""

JUDGE_SYSTEM_PROMPT = """
אתה שופט שמעריך תשובות של מערכת AI.
קבלת שאלה, תשובה צפויה ותשובה שהמערכת החזירה.
עליך לדרג את התשובה בסקלה של 1-10 ולהסביר בקצרה.
החזר תשובה בפורמט JSON בלבד:
{
  "score": <מספר בין 1 ל-10>,
  "reason": "<הסבר קצר בעברית>"
}
"""


def run_agent(question: str) -> str:
    """שולח שאלה לאייג'נט ומחזיר תשובה."""
    documents = retrieve_relevant_chunks(question, 2)
    documents_prompt = f"""
    הקבצים המצורפים הם:

{chr(10).join(doc['text'] for doc in documents)}
"""
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "system", "content": documents_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def judge_answer(question: str, expected: str, actual: str) -> dict:
    """שולח לLLM שופט ומקבל ציון."""
    user_prompt = f"""
שאלה: {question}
תשובה צפויה: {expected}
תשובה שהתקבלה: {actual}
"""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "שגיאה בפענוח תשובת השופט"}


def run_evaluation():
    dataset = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))

    results = []

    for item in dataset:
        print(f"[{item['id']}] {item['question']}")

        actual_answer = run_agent(item["question"])
        judgment = judge_answer(item["question"], item["expected_answer"], actual_answer)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "actual_answer": actual_answer,
            "score": judgment["score"],
            "reason": judgment["reason"],
        })

        print(f"  ציון: {judgment['score']}/10 — {judgment['reason']}\n")

    # שמירה ל-CSV
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n✅ סיום Evaluation")
    print(f"📊 ציון ממוצע: {avg:.1f}/10")
    print(f"📁 תוצאות נשמרו ב: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()