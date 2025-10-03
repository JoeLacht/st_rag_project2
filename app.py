import streamlit as st
import numpy as np
import joblib
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from symspellpy import SymSpell, Verbosity
from sentence_transformers import CrossEncoder
import os, getpass
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

st.cache_data.clear()
st.cache_resource.clear()

# --- Настройки страницы ---
st.set_page_config(
    page_title="🎥 Семантический поиск фильмов",
    page_icon="🎬",
    layout="wide"
)
st.title("🎬 Семантический поиск фильмов")
st.markdown("Введите запрос, и я подберу подходящие фильмы с юмором 🤓✨")

# --- Загрузка эмбеддингов и Faiss ---
embeddings = np.load("data/movie_embeds.npy")
meta = joblib.load("data/movie_meta.pkl")
index = faiss.read_index("data/faiss_index.bin")

dim = embeddings.shape[1]

# --- Embeddings модель ---
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
)

# --- SymSpell для исправления опечаток ---
from symspellpy import SymSpell, Verbosity

max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

# функция безопасного split
def safe_split(s, sep=' '):
    return str(s).split(sep) if isinstance(s, str) else []

# создаём словарь для SymSpell
vocab = set()
for m in meta:
    for w in safe_split(m.get("title_ru")):
        vocab.add(w.lower())
    for w in safe_split(m.get("genres"), sep=','):
        vocab.add(w.lower())

# загружаем в symspell
for w in vocab:
    sym_spell.create_dictionary_entry(w, 1)

# функция исправления запроса
def symspell_correct_query(query: str):
    words = query.split()
    corrected = []
    for w in words:
        suggestions = sym_spell.lookup(w.lower(), Verbosity.TOP, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(w)
    return " ".join(corrected)

# --- CrossEncoder для rerank ---
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# --- Функция поиска ---
def search(query, top_k_faiss=50, top_k_return=10, apply_symspell=True):
    q0 = query
    if apply_symspell:
        q = symspell_correct_query(query)
    else:
        q = query

    q_emb = embeddings_model.embed_query(q)
    q_emb = np.array(q_emb, dtype=np.float32).reshape(1, -1)

    D, I = index.search(q_emb, top_k_faiss)
    candidate_idxs = I[0].tolist()

    candidates = []
    for idx in candidate_idxs:
        if idx < 0: continue
        candidates.append((meta[idx]['title_ru'], idx))

    cross_inputs = [[q0, meta[idx]['overview_ru']] for _, idx in candidates]
    rerank_scores = cross_encoder.predict(cross_inputs)

    ranked = sorted(zip([idx for _, idx in candidates], rerank_scores), key=lambda x: x[1], reverse=True)
    top_ranked = ranked[:top_k_return]

    results = []
    for idx, score in top_ranked:
        m = meta[idx]
        results.append({
            'score': float(score),
            'idx': idx,
            'title_ru': m.get('title_ru'),
            'meta.year': m.get('meta.year'),
            'meta.tmdb_id': m.get('meta.tmdb_id'),
            'meta.poster_url': m.get('meta.poster_url'),
            'directors': m.get('directors'),
            'actors_main': m.get('actors_main'),
            'genres': m.get('genres'),
            'overview_ru': m.get('overview_ru')
        })
    return results

# --- UI ---
query = st.text_input("🔍 Введите запрос:", "космос, комедия")
k_value = st.slider("📊 Сколько фильмов показать:", 1, 10, 3)
temperature_value = st.slider(
    "🔥 Креативность ответа (temperature):", 
    min_value=0.0, 
    max_value=1.5, 
    value=0.7, 
    step=0.05
)
apply_spell = st.checkbox("Исправлять опечатки", True)

# --- Настройка LLM ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=temperature_value, max_tokens=4096)

SYSTEM_PROMPT = """Ты кинокритик с многолетним опытом и отличным чувством юмора! 🎯
    Твоя задача - проанализировать предоставленные фильмы и дать профессиональную оценку с долей иронии.

    Стиль анализа:
    - Проводи глубокий анализ, развернутые описания, рекомендации, связи между произведениями, но с легкой иронией
    - Используй кино-мемы и шутки там, где это уместно
    - Структурируй ответ с эмодзи и забавными комментариями
    - Отвечай на русском языке живым, но профессиональным тоном

    Помни: юмор должен быть добрым и не оскорбительным. Цель - сделать анализ интересным!

    Если среди фильмов есть что-то особенно забавное - обязательно это отметь! 😄"""

def build_rag_context(results, user_query, max_chars=1800):
    parts = []
    for r in results:
        s = f"Title: {r['title_ru']} ({r.get('meta.year','')})\nGenres: {r.get('genres','')}\nDescription: {r.get('overview_ru')[:200]}\nURL: {r.get('meta.tmdb_id')}\n"
        parts.append(s)
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars]
    prompt = f"{SYSTEM_PROMPT}\n\nUser query: {user_query}\n\nContext:\n{context}\n\nОтвет:"
    return prompt

def rag_answer(user_query, results):
    prompt = build_rag_context(results, user_query)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {user_query}\n\nContext:\n{build_rag_context(results, user_query)}")
    ]
    resp = llm(messages)
    return resp.content

import re

def clean_think_tags(text: str) -> str:
    # удаляем всё между <think> и </think> (включительно)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

if st.button("Найти 🎯") and query.strip():
    with st.spinner("🚀 Подбираем фильмы..."):
        results = search(query, top_k_faiss=50, top_k_return=k_value, apply_symspell=apply_spell)
        # rag_resp = rag_answer(query, results)
        # # st.subheader("🎭 Анализ кинокритика:")
        # # st.markdown(rag_resp)
        rag_resp = rag_answer(query, results)
        rag_resp_clean = clean_think_tags(rag_resp)
        st.subheader("🎭 Анализ кинокритика:")
        st.markdown(rag_resp_clean)

        st.subheader("🎬 Найденные фильмы:")
        for r in results:
            col1, col2 = st.columns([1,3])
            with col1:
                if r.get('meta.poster_url'):
                    st.image(r["meta.poster_url"], use_container_width=True)
            with col2:
                st.markdown(f"### [{r.get('title_ru','Не указано')}]({r.get('meta.poster_url','#')})")
                st.markdown(f"**Жанры:** {r.get('genres','Не указано')}")
                st.markdown(f"**Режиссер:** {r.get('directors','Не указано')}")
                st.markdown(f"**Актёры:** {r.get('actors_main','Не указано')}")
                st.markdown(f"**Год:** {r.get('meta.year','Не указано')}")
                movie_url = r.get('meta.tmdb_id', '#')
                st.markdown(f"**[Ссылка на фильм]({movie_url})**")
                st.markdown(f"**Описание:** {r.get('overview_ru')}...")
            st.markdown("---")
