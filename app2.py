import os
import sys
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from dotenv import load_dotenv

load_dotenv()

# הגדרות דף בסיסיות של Streamlit
st.set_page_config(page_title="Biology Agentic Chat", page_icon="💬", layout="centered")

# הגרת כיוון דף לעברית (RTL) כולל רכיבי הצ'אט החדשים
st.markdown("""
    <style>
    .stApp { direction: rtl; text-align: right; }
    div[data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    div[data-testid="stExpander"] div { direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("💬 צ'אט ביולוגיה אינטראקטיבי - 3 סוכנים")
st.subheader("נהל שיחה רציפה עם ספר הלימוד שלך")

# הגדרות נתיבים ומפתחות (תעדכן אותם אחי!)
db_path = r"C:\Users\boaza\Downloads\chroma_db\chroma_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def init_rag_system():
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    client = genai.Client(api_key=GEMINI_API_KEY)
    return db, client

try:
    db, client = init_rag_system()
except Exception as e:
    st.error(f"שגיאה בחיבור למערכת: {e}")

# אתחול היסטוריית הצ'אט בזיכרון של הדף (אם היא עדיין לא קיימת)
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת כל הודעות העבר על המסך בכל רענון של הדף
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# תיבת קלט של הצ'אט בתחתית המסך (כמו ב-ChatGPT)
if user_query := st.chat_input("שאל אותי משהו על ספר הביולוגיה..."):
    
    # 1. הצגת שאלת המשתמש במסך בצורה מיידית
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # שמירת השאלה בהיסטוריה
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. תהליך הפעלת 3 הסוכנים
    with st.chat_message("assistant"):
        with st.spinner("⏳ צוות הסוכנים חושב ומנתח את ההיסטוריה..."):
            try:
                # בניית פורמט היסטוריה מקוצר עבור הסוכנים
                chat_history_str = ""
                for msg in st.session_state.messages[:-1]: # לוקח את השיחה עד השאלה הנוכחית
                    role_label = "סטודנט" if msg["role"] == "user" else "מערכת"
                    chat_history_str += f"{role_label}: {msg['content']}\n"

                # --- סוכן 1: החוקר (לוקח בחשבון את ההיסטוריה!) ---
                search_prompt = f"""
                נתונה היסטוריית השיחה הבאה בביולוגיה:
                {chat_history_str}
                
                והנה השאלה החדשה של הסטודנט: '{user_query}'.
                בהתבסס על כל השיחה, כתוב אך ורק את מונחי המפתח המדעיים המדויקים ביותר (בשלשות או זוגות מילים) שצריך לשלוף מספר הלימוד כדי לענות על השאלה החדשה. אל תוסיף מילים מעבר למונחים.
                """
                search_refinement = client.models.generate_content(model='gemini-2.5-flash', contents=search_prompt).text.strip()
                
                # שליפה מהמאגר
                docs = db.similarity_search(search_refinement, k=5)
                context = "\n\n".join([f"--- מקטע {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)])

                # --- סוכן 2: המרצה ---
                writer_instruction = "אתה מרצה לביולוגיה באוניברסיטה. תפקידך לנסח תשובה מפורטת, מדעית ומובנית היטב על בסיס מקטעי הטקסט המצורפים בלבד. השתמש בשפה אקדמית גבוהה וקח בחשבון את זרימת השיחה הקודמת במידת הצורך."
                writer_prompt = f"היסטוריית השיחה:\n{chat_history_str}\n\nהמקטעים שנשלפו מהספר:\n{context}\n\nענה על השאלה החדשה: {user_query}"
                draft_response = client.models.generate_content(
                    model='gemini-2.5-flash', contents=writer_prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=writer_instruction, temperature=0.3)
                ).text

                # --- סוכן 3: המבקר ---
                critic_instruction = "אתה בודק בחינות קשוח באוניברסיטה. תפקידך לקחת את טיוטת התשובה, להשוות אותה למקטעי המקור בלבד, ולתקן אותה. ודא שאין המצאות, שכל חלקי השאלה נענו, ושהניסוח מושלם."
                critic_prompt = f"מקטעי המקור מהספר:\n{context}\n\nהשאלה המקורית:\n{user_query}\n\nטיוטת המרצה:\n{draft_response}\n\nאנא ספק את התשובה הסופית, המשופרת והמתוקנת ביותר בעברית:"
                final_response = client.models.generate_content(
                    model='gemini-2.5-flash', contents=critic_prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=critic_instruction, temperature=0.1)
                ).text

                # הצגת התשובה הסופית בצ'אט
                st.markdown(final_response)
                
                # הצגת קופסת מידע מתקפלת "מאחורי הקלעים" מתחת לתשובה
                with st.expander("🛠️ הצץ לעבודת הסוכנים מאחורי הקלעים של הודעה זו"):
                    st.write(f"**🔍 שאילתת החיפוש שהחוקר ייצר:** `{search_refinement}`")
                    st.text(f"📚 תוכן המקטעים שנשלפו מהספר:\n{context}")
                    st.markdown(f"📝 הטיוטה הראשונית של המרצה:\n{draft_response}")

                # שמירת תשובת האסיסטנט בהיסטוריה
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"חטפנו שגיאה בריצה: {e}")