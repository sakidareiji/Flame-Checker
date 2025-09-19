import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI()

class CheckRequest(BaseModel):
    post: str

class CheckResponse(BaseModel):
    risk_level: Literal["high", "middle", "low"]
    ai_comment: str

@app.get("/")
def read_root():
    return {"message": "API is running!"}


@app.post("/check/post", response_model=CheckResponse)
def check_post_endpoint(request: CheckRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    post_text = request.post

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt = f"""
        あなたはSNS投稿のリスク分析を専門とするエキスパートです。
        以下の投稿内容を分析し、炎上するリスクの高さを判定してください。
        判定結果は、必ず以下のJSON形式で返してください。

        {{
          "risk_level": "lowかmiddleかhighのいずれか",
          "ai_comment": "判定理由を30文字程度で具体的に記述"
        }}

        ---
        【投稿内容】
        {post_text}
        """

        response = model.generate_content(prompt)
        cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        result_data = json.loads(cleaned_text)

        risk_level = result_data.get("risk_level", "low")
        ai_comment = result_data.get("ai_comment", "分析に失敗しました。")

        if risk_level not in ["high", "middle", "low"]:
            risk_level = "low"

        return CheckResponse(
            risk_level=risk_level,
            ai_comment=ai_comment
        )

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse Gemini response: {e}")
        raise HTTPException(
            status_code=500,
            detail="AIからのレスポンス形式が不正です。"
        )
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(
            status_code=503,
            detail="外部AIサービスで一時的な問題が発生しました。"
        )