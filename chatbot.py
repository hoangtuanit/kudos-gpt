import os
import json
import random
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import os
import pickle
import openai
from openai import AsyncOpenAI
import asyncio
import faiss
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Load biến môi trường
load_dotenv()


# Flask API
app = Flask(__name__)
CORS(app)  # Cho phép tất cả domain gọi API

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

# # Đọc file CSV chứa các review
df = pd.read_csv('amazon-reviews/cleaned_reviews.csv')

# # Lấy danh sách review, loại bỏ NaN và chuyển về str
reviews = df['cleaned_review'].dropna().astype(str).tolist()
## Limit first 100 reviews for testing
reviews = reviews[:100]

NLTK_RES_PATH = "nltk_resources.pkl"

def prepare_nltk_resources():
    if os.path.exists(NLTK_RES_PATH):
        with open(NLTK_RES_PATH, "rb") as f:
            stop_words, lemmatizer = pickle.load(f)
    else:
        nltk.download('stopwords', download_dir='./nltk_data')
        nltk.download('wordnet', download_dir='./nltk_data')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        with open(NLTK_RES_PATH, "wb") as f:
            pickle.dump((stop_words, lemmatizer), f)
    return stop_words, lemmatizer

# Sử dụng:
stop_words, lemmatizer = prepare_nltk_resources()

def preprocess_review(review):
    # Loại bỏ ký tự đặc biệt và số
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    # Chuyển về chữ thường
    review = review.lower()
    # Tách từ
    words = review.split()
    # Loại bỏ stopwords và lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Ghép lại thành câu
    return ' '.join(words)

# # Tiền xử lý các review
reviews = [preprocess_review(r) for r in reviews]


# # Lấy vector từ OpenAI
def get_embedding(text):
    print(f"Generating embedding for text: {text[:30]}...")  # Log first 30 chars
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )

    return np.array(response.data[0].embedding)


def search_reviews(query, top_k=3):
    q_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    ## in ra khoảng cách và chỉ mục
    return [reviews[i] for i in indices[0]]

def generate_response(product_info, reviews, user_question):
    system_prompt = f"""
You are a product assistant. You will answer customer questions based on product specifications and real customer reviews.

Product Specs:
- Name: {product_info['name']}
- Size: {product_info['width_cm']}cm x {product_info['height_cm']}cm x {product_info['depth_cm']}cm
- Material: {product_info['material']}
- Features: {", ".join(product_info['features'])}

Customer Reviews (summarized):
{reviews}

Customer Question:
"{user_question}"

Answer in helpful and natural language. Be clear and honest.
"""
    response = openai.chat.completions.create(
        # model="gpt-4",
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content



FAISS_INDEX_PATH = "faiss.index"
EMBEDDINGS_PATH = "embeddings.npy"

def save_faiss_index(index, embeddings):
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        return index, embeddings
    return None, None

def build_faiss_index(reviews):
    review_embeddings = np.array([get_embedding(review) for review in reviews])
    dim = review_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(review_embeddings)
    # Lưu lại index và embeddings
    save_faiss_index(index, review_embeddings)
    return index, review_embeddings

# Khởi tạo biến toàn cục cho index và embeddings
index, review_embeddings = load_faiss_index()

def ensure_faiss_index():
    global index, review_embeddings
    if index is None or review_embeddings is None:
        print("Building FAISS index...")
        index, review_embeddings = build_faiss_index(reviews)
        print("FAISS index built and saved.")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    print("Received data:", data, "\n")
    product_info = data.get('product_info')
    user_question = data.get('user_question')
    if not product_info or not user_question:
        return jsonify({'error': 'Missing product_info or user_question'}), 400
    # Đảm bảo FAISS index đã sẵn sàng
    ensure_faiss_index()
    # Tìm các review liên quan nhất
    relevant_reviews = search_reviews(user_question, top_k=1)
    # Sinh câu trả lời
    answer = generate_response(product_info, relevant_reviews, user_question)
    return jsonify({'answer': answer})

@app.route('/chat-stream', methods=['POST'])
def chat_stream():
    data = request.json
    product_info = data.get('product_info')
    user_question = data.get('user_question')
    if not product_info or not user_question:
        return jsonify({'error': 'Missing product_info or user_question'}), 400
    ensure_faiss_index()
    relevant_reviews = search_reviews(user_question, top_k=1)
    system_prompt = f"""
You are a product assistant. You will answer customer questions based on product specifications and real customer reviews.

Product Specs:
- Name: {product_info['name']}
- Size: {product_info['width_cm']}cm x {product_info['height_cm']}cm x {product_info['depth_cm']}cm
- Material: {product_info['material']}
- Features: {", ".join(product_info['features'])}

Customer Reviews (summarized):
{relevant_reviews}

Customer Question:
"{user_question}"

Answer in helpful and natural language. Be clear and honest.
"""

    def generate():
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            stream=True
        )
        for chunk in response:
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                yield content.encode("utf-8")

    return Response(generate(), mimetype='text/plain')

# # Tạo index FAISS
# review_embeddings = [get_embedding(review) for review in reviews]
# index = faiss.IndexFlatL2(len(review_embeddings[0]))
# index.add(np.array(review_embeddings))

api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

if __name__ == '__main__':
    # Build hoặc load FAISS index trước khi start Flask
    ensure_faiss_index()
    # Thêm dòng này để chỉ định đường dẫn cho nltk data
    
    nltk.data.path.append('./nltk_data')
    app.run(debug=True, port=5000, host='0.0.0.0')