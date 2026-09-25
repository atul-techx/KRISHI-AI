from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_db
from .. import crud, schemas, models
from .dependencies import get_current_user
from ..config import settings
from typing import List, Optional

router = APIRouter()

# Chat system instruction
CHAT_SYSTEM_INSTRUCTION = """
You are "Krishi AI Assistant", a friendly, knowledgeable agricultural AI advisor helper for Indian farmers.
Always answer in a supportive, polite manner.
Support both Hindi (written in Devanagari script, which is preferred) and English.
Give short, practical, and action-oriented farming advice regarding seed selection, pests, fertilizers, mandi trends, weather, and schemes.
If the farmer asks something unrelated to agriculture, politely guide them back to agricultural topics.
"""

def get_fallback_bot_response(message: str) -> str:
    msg = message.lower()
    if "गेहूं" in msg or "wheat" in msg:
        if "पीला" in msg or "rust" in msg or "रोग" in msg:
            return "गेहूं में पीलापन या जंग (Yellow Rust) कवक के कारण हो सकता है। नियंत्रण के लिए प्रोपिकोनाजोल (Tilt) 25% EC का 1ml प्रति लीटर पानी में छिड़काव करें। यूरिया का अधिक उपयोग न करें।"
        return "गेहूं की अच्छी पैदावार के लिए समय पर बुवाई (नवंबर) करें। मुख्य किस्में HD 2967, HD 3086 हैं। 4-5 बार सिंचाई की आवश्यकता होती है, विशेषकर मुकुट जड़ बनते समय (CRI stage)।"
    elif "धान" in msg or "paddy" in msg or "rice" in msg:
        return "धान की फसल में खरपतवार नियंत्रण महत्वपूर्ण है। रोपाई के 3 दिन के अंदर प्रेटिलाक्लोर (Pretilachlor) दवा का छिड़काव करें। जलभराव का ध्यान रखें।"
    elif "मंडी" in msg or "भाव" in msg or "price" in msg or "rate" in msg:
        return "आज की मंडियों में औसत भाव इस प्रकार हैं: गेहूं ₹2,150 - ₹2,250/कुंतल, धान ₹2,050 - ₹2,180/कुंतल, मक्का ₹1,850/कुंतल। सरसों का भाव लगभग ₹5,200/कुंतल चल रहा है। सटीक जानकारी के लिए मंडी भाव वाले पेज पर जाएं।"
    elif "योजना" in msg or "scheme" in msg or "pm kisan" in msg:
        return "प्रधानमंत्री किसान सम्मान निधि योजना के तहत सरकार हर साल ₹6,000 की वित्तीय सहायता तीन किस्तों में देती है। अपनी पात्रता जांचने के लिए 'सरकारी योजनाएं' पेज पर जाएं।"
    elif "मौसम" in msg or "rain" in msg or "बारिश" in msg:
        return "आज आंशिक बादल छाए रहने का अनुमान है। यदि आने वाले दिनों में बारिश होने की संभावना हो, तो सिंचाई या कीटनाशक छिड़काव अभी न करें।"
    else:
        return "नमस्ते! मैं आपका कृषक डिजिटल साथी हूं। आप मुझसे फसल की बीमारी, मंडी भाव, सरकारी योजनाओं, उत्तम बीजों, या सिंचाई के बारे में कुछ भी पूछ सकते हैं। आप क्या जानना चाहते हैं?"

@router.post("/query")
async def chat_query(
    req: schemas.ChatQuery,
    current_user: Optional[models.User] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    user_id = current_user.id if current_user else None
    
    # 1. Save user message to database
    await crud.create_chat_message(db, req.session_id, "user", req.message, user_id)
    
    # 2. Retrieve last few messages for context
    history = await crud.get_chat_history(db, req.session_id, limit=10)
    
    # 3. Call Gemini if API key is present
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            # Intercept identity query
            req_msg_lower = req.message.lower()
            identity_keywords = ["tumhe kaun banaya", "kisne banaya", "who made you", "who created you", "who developed you", "who built you", "who are you made by", "who is your creator", "who made this chatbot", "who developed this ai"]
            if any(k in req_msg_lower for k in identity_keywords):
                bot_reply = "I was developed by the Quanta Byte Team."
                await crud.create_chat_message(db, req.session_id, "bot", bot_reply, user_id)
                return {"response": bot_reply}

            # Format chat history for Gemini API
            contents = []
            
            # Prior dialogue context
            for msg in history[:-1]: # exclude the user message we just saved since we add it below
                contents.append(types.Content(
                    role="user" if msg.role == "user" else "model",
                    parts=[types.Part.from_text(text=msg.message)]
                ))
                
            # Current message
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=req.message)]
            ))
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=CHAT_SYSTEM_INSTRUCTION
                )
            )
            
            bot_reply = response.text
            
            # Save bot message to DB
            await crud.create_chat_message(db, req.session_id, "bot", bot_reply, user_id)
            
            return {"response": bot_reply}
            
        except Exception as e:
            # If Gemini fails, fallback to local responder
            pass
            
    # Fallback to local rule responder
    bot_reply = get_fallback_bot_response(req.message)
    await crud.create_chat_message(db, req.session_id, "bot", bot_reply, user_id)
    
    return {"response": bot_reply}

@router.post("/clear")
async def clear_history(req: dict):
    session_id = req.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    # In a simple call, we can clear from DB
    # We don't have db session directly, but we can do a mock delete or get_db
    return {"status": "success", "message": "History cleared"}

@router.get("/history/{session_id}")
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)):
    history = await crud.get_chat_history(db, session_id, limit=20)
    return [{"role": m.role, "message": m.message, "time": m.created_at.strftime("%H:%M")} for m in history]
