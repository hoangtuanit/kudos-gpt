import os
import json
import random
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import asyncio
import faiss
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory
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

NLTK_RES_PATH = "./nltk/nltk_resources.pkl"

def prepare_nltk_resources():
    nltk.data.path.append('./nltk_data')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Sử dụng:
stop_words, lemmatizer = prepare_nltk_resources()

def preprocess_review(review):
     # Loại bỏ ký tự đặc biệt, giữ lại chữ, số, dấu câu cơ bản
    review = re.sub(r"[^a-zA-ZÀ-ỹ0-9,.!?;:\s\-]", '', review)
    # Chuyển về chữ thường
    review = re.sub(r'https?://\S+|www\.\S+', '', review)
    review = re.sub(r'<.*?>', '', review)
    # Rút gọn khoảng trắng
    review = re.sub(r'\s+', ' ', review).strip()
    # Tách từ
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
        input=text  # Pass as string, not list
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

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# --- SUMMARIZATION API ---
import asyncio

async def summarize_features_for_buyer(reviews):
    prompt = f"""
        You will be given a batch of product reviews. Summarize them by extracting:

        - Overall sentiment (positive, negative, mixed)
        - Common PROS: list 3–5 things customers often praised
        - Common CONS: list 2–3 frequent complaints
        - Optional: 1–2 short, representative quotes
        - Final verdict: Should a potential buyer feel confident in the product?

        Here are the reviews:
        ---
        {reviews}
        ---
    """
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()

async def summarize_features_for_merchant(reviews):
    prompt = f"""
        You are an assistant for a business owner who wants to improve their product based on customer feedback.

Given a list of customer reviews, summarize the most common positive and negative opinions. Group the feedback into key aspects (e.g., battery, design, performance, price, etc.). Then, give clear suggestions to improve the product based on customer concerns.

Reviews:
        ---
        {reviews}
        ---
    """
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()

def parse_summary_from_text(text):
    # Đơn giản: tách các mục dựa trên tiêu đề thường gặp
    result = {"pros": [], "cons": [], "sentiment": None, "quotes": [], "verdict": None}
    # Sentiment
    sentiment_match = re.search(r"Overall sentiment\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if sentiment_match:
        result["sentiment"] = sentiment_match.group(1).strip()
    # Pros
    pros_match = re.search(r"Common PROS\s*[:\-]?\s*(.+?)(?:Common CONS|Cons|\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if pros_match:
        pros_text = pros_match.group(1)
        result["pros"] = [p.strip('-•* \n') for p in pros_text.strip().split('\n') if p.strip('-•* ')]
    # Cons
    cons_match = re.search(r"Common CONS\s*[:\-]?\s*(.+?)(?:Optional|\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if not cons_match:
        cons_match = re.search(r"Cons\s*[:\-]?\s*(.+?)(?:Optional|\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if cons_match:
        cons_text = cons_match.group(1)
        result["cons"] = [c.strip('-•* \n') for c in cons_text.strip().split('\n') if c.strip('-•* ')]
    # Quotes
    quotes_match = re.search(r"quotes?\s*[:\-]?\s*(.+?)(?:Final verdict|\n\n|$)", text, re.IGNORECASE | re.DOTALL)
    if quotes_match:
        quotes_text = quotes_match.group(1)
        quotes = re.findall(r'"([^"]+)"', quotes_text)
        if not quotes:
            quotes = [q.strip('-•* \n') for q in quotes_text.strip().split('\n') if q.strip('-•* ')]
        result["quotes"] = quotes
    # Verdict
    verdict_match = re.search(r"Final verdict\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).strip()
    return result

@app.route('/summary-buyer', methods=['POST'])
def summary_buyer():
    data = request.json
    reviews_input = data.get('reviews')
    if not reviews_input:
        reviews_input = reviews
    if isinstance(reviews_input, list):
        reviews_text = '\n'.join(reviews_input)
    else:
        reviews_text = str(reviews_input)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    summary_text = loop.run_until_complete(summarize_features_for_buyer(reviews_text))
    summary = parse_summary_from_text(summary_text)
    return jsonify({'summary': summary})

@app.route('/summary-merchant', methods=['POST'])
def summary_merchant():
    data = request.json
    reviews_input = data.get('reviews')
    if not reviews_input:
        reviews_input = reviews
    if isinstance(reviews_input, list):
        reviews_text = '\n'.join(reviews_input)
    else:
        reviews_text = str(reviews_input)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    summary_text = loop.run_until_complete(summarize_features_for_merchant(reviews_text))
    summary = parse_summary_from_text(summary_text)
    return jsonify({'summary': summary})

# # Tạo index FAISS
# review_embeddings = [get_embedding(review) for review in reviews]
# index = faiss.IndexFlatL2(len(review_embeddings[0]))
# index.add(np.array(review_embeddings))

api_key = os.getenv("OPENAI_API_KEY")
debug_mode = os.getenv("DEBUG_MODE")
client = AsyncOpenAI(api_key=api_key)

if __name__ == '__main__':
    ensure_faiss_index()
    nltk.data.path.append('./nltk_data')
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)