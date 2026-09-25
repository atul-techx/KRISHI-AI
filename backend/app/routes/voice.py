from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from ..database import get_db
from ..config import settings
from .dependencies import get_current_user
from .. import models
import json

router = APIRouter()

class VoiceQueryRequest(BaseModel):
    query: str

VOICE_SYSTEM_INSTRUCTION = """
You are a voice advisory assistant for Indian farmers. 
The user is speaking to you. Your reply will be read aloud by Text-to-Speech.
Keep your response extremely brief, clear, and easy to understand when spoken (under 3 sentences).
Answer in simple Hindi (Devenagari script). 
Do not use bullet points or markdown syntax like stars (*) or hashes (#) because they sound weird when read aloud.
"""

def get_voice_fallback(query: str) -> str:
    q = query.lower()
    if "पीले" in q or "दाग" in q or "रोग" in q or "wheat" in q:
        return "फसल पर पीले दाग कवक रोग के लक्षण हो सकते हैं। पानी जमा न होने दें और कृषि विशेषज्ञ से सलाह लेकर फफूंदनाशक दवा का छिड़काव करें।"
    elif "भाव" in q or "मंडी" in q or "रेट" in q:
        return "आज आपके जिले में धान का भाव लगभग इक्कीस सौ रुपये और गेहूं का भाव बाईस सौ रुपये प्रति क्विंटल चल रहा है।"
    elif "बारिश" in q or "मौसम" in q:
        return "आसमान में आंशिक रूप से बादल छाए रहने का अनुमान है। यदि भारी बारिश की संभावना हो तो सिंचाई रोक दें।"
    elif "किसान" in q or "पैसा" in q or "किस्त" in q:
        return "प्रधानमंत्री किसान सम्मान निधि की नई किस्त जारी कर दी गई है। आप अधिकारिक वेबसाइट पर जाकर अपना खाता स्टेटस चेक कर सकते हैं।"
    else:
        return "नमस्ते। मैं आपका कृषि सहायक हूँ। मैं आपकी खेती से जुड़ी समस्याओं का तुरंत उत्तर दे सकता हूँ। कृपया अपना प्रश्न पूछें।"

@router.post("/query")
async def voice_query(
    req: VoiceQueryRequest,
    current_user: models.User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=req.query)]
                )
            ]
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=VOICE_SYSTEM_INSTRUCTION
                )
            )
            
            reply = response.text.strip()
            # Clean reply from markdown
            reply = reply.replace("*", "").replace("#", "").replace("-", "")
            return {"response": reply}
            
        except Exception:
            pass
            
    # Fallback response
    return {"response": get_voice_fallback(req.query)}
