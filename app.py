import streamlit as st
import torch
import joblib
import numpy as np
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- הגדרות עיצוב לדף ---
st.set_page_config(page_title="זיהוי פנים - פרויקט גמר", page_icon="🎭", layout="centered")

# 💡 הוספת קוד עיצוב (CSS) שהופך את התצוגה החיה של המצלמה ל"מראה"
st.markdown(
    """
    <style>
    [data-testid="stCameraInput"] video {
        transform: scaleX(-1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎭 מערכת לזיהוי פנים")
st.write("פרויקט גמר בבינה מלאכותית. העלו תמונה או צלמו כדי לזהות מי בתמונה!")

# --- טעינת מודלים ---
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model = joblib.load('face_model.pkl')
    return device, mtcnn, resnet, model

device, mtcnn, resnet, model = load_models()

# רשימת השמות
CLASS_NAMES = ['asaf', 'danny', 'eitan', 'harel', 'ilay', 'kiril', 'lior', 
               'ofir', 'ofri', 'omri', 'rotem', 'segev', 'semyon', 'yuval']

# --- בחירת תמונה עם טאבים ---
tab1, tab2 = st.tabs(["📸 הפעל מצלמה", "📂 העלאת קובץ מגלריה"])

image_file = None
is_camera = False # משתנה שזוכר מאיפה הגיעה התמונה

with tab1:
    camera_img = st.camera_input("צלמו תמונה כאן")
    if camera_img:
        image_file = camera_img
        is_camera = True # סימון שהתמונה הגיעה מהמצלמה

with tab2:
    uploaded_img = st.file_uploader("בחרו תמונה", type=['jpg', 'jpeg', 'png'])
    if uploaded_img:
        image_file = uploaded_img
        is_camera = False # סימון שהתמונה הגיעה מהגלריה

# --- תהליך הזיהוי ---
if image_file is not None:
    # פתיחת התמונה
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img) # תיקון סיבוב 90 מעלות
    
    # 💡 היפוך התמונה (אפקט מראה) יבוצע רק אם התמונה צולמה במצלמה
    if is_camera:
        img = ImageOps.mirror(img)
        
    img = img.convert('RGB')
    
    st.image(img, caption="התמונה שהוזנה", use_container_width=True)
    
    with st.spinner("מנתח את התמונה... 🔍"):
        # 1. זיהוי וחיתוך פנים
        face_tensor, prob = mtcnn(img, return_prob=True)
        
        if face_tensor is not None and prob > 0.85:
            # 2. חילוץ מאפיינים (Embeddings)
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(face_tensor).cpu().numpy()
            
            # 3. חיזוי
            prediction_idx = model.predict(emb)[0]
            prediction_name = CLASS_NAMES[prediction_idx]
            
            probabilities = model.predict_proba(emb)[0]
            confidence = probabilities[prediction_idx] * 100
            
            # 4. הצגת התוצאה למשתמש
            st.success("זיהוי מוצלח! 🎉")
            st.markdown(f"<h2 style='text-align: center;'>האיש בתמונה: {prediction_name.capitalize()}</h2>", unsafe_allow_html=True)
            st.info(f"רמת ביטחון: {confidence:.2f}%")
        else:
            st.error("לא הצלחתי לזהות פנים ברורות בתמונה. נסו לצלם שוב עם תאורה טובה יותר.")
