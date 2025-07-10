from flask import Flask, request, jsonify, render_template
import wikipedia
from duckduckgo_search import DDGS
from transformers import pipeline

app = Flask(__name__)
generator = pipeline("text-generation", model="gpt2")  # Replace with larger model if needed

def get_wiki_content(query):
    try:
        summary = wikipedia.summary(query, sentences=5)
        return summary
    except Exception:
        return ""

def web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join(r["body"] for r in results if "body" in r)
    except Exception:
        return ""

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")

    wiki_data = get_wiki_content(question)
    web_data = web_search(question)

    full_context = wiki_data + "\n" + web_data
    prompt = f"Answer the question based on the following information:\n\n{full_context}\n\nQuestion: {question}\nAnswer:"

    result = generator(prompt, max_length=100, temperature=0.7)[0]['generated_text']
    answer = result.split("Answer:")[-1].strip()

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
