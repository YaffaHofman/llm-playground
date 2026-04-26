import os
import base64
from typing import Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# =====================
# הגדרות בסיסיות
# =====================
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".pdf"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================
# מודל נתונים
# =====================
class PersonID(BaseModel):
    full_name: str
    id_number: str


# =====================
# בדיקות קובץ
# =====================
def is_valid_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# =====================
# המרה ל־base64
# =====================
def encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =====================
# קריאה ל־OpenAI (Vision)
# =====================
def extract_data(file_path: str) -> Optional[PersonID]:
    try:
        base64_file = encode_file(file_path)

        response = client.responses.parse(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract full name and ID number from this document. Return only valid data."
                    },
                    {
                        "type": "input_image",
                        "image_base64": base64_file
                    }
                ]
            }],
            response_format=PersonID
        )

        return response.output_parsed

    except ValidationError:
        return None
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        return None


# =====================
# Tool – שמירה
# =====================
def save_user(full_name: str, id_number: str):
    with open("users.txt", "a", encoding="utf-8") as f:
        f.write(f"{full_name},{id_number}\n")


# =====================
# ולידציה בסיסית
# =====================
def validate_id(id_number: str) -> bool:
    return id_number.isdigit() and len(id_number) in [8, 9]


# =====================
# לוגיקה ראשית
# =====================
def main():
    print("📄 העלי תעודת זהות / דרכון / רישיון נהיגה")

    while True:
        file_path = input("📥 הכניסי נתיב קובץ: ").strip()

        # בדיקת קובץ
        if not is_valid_file(file_path):
            print("❌ קובץ לא תקין. נסי שוב (jpg/png/pdf)")
            continue

        print("🔍 מנתח את הקובץ...")

        data = extract_data(file_path)

        if not data:
            print("⚠️ לא הצלחנו לזהות נתונים. נסי קובץ ברור יותר")
            continue

        # ולידציה
        if not validate_id(data.id_number):
            print("⚠️ מספר תעודת זהות לא תקין")
            continue

        # שמירה
        save_user(data.full_name, data.id_number)

        print("✅ נשמר בהצלחה!")
        print(f"👤 שם: {data.full_name}")
        print(f"🆔 ת\"ז: {data.id_number}")

        break


# =====================
# הרצה
# =====================
if __name__ == "__main__":
    main()