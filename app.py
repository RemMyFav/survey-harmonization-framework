from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for

from generator import (
    SeededQuestionGenerator,
    retrieve_seeds,
    seed_texts_from_df,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

DIMENSIONS = [
    "Emotional",
    "Environmental",
    "Financial",
    "Intellectual",
    "Occupational",
    "Physical",
    "Social",
    "Spiritual",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SOURCE_FILE = os.path.join(DATA_DIR, "source_questions.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

BASE_STYLE = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
    background: #ffffff; 
    color: #1a1a1a; 
    min-height: 100vh; 
}
.container { 
    width: 100%; 
    max-width: 640px; 
    margin: 0 auto;
    padding: 60px 20px 20px;
}
"""

WELCOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Item Generator</title>
    <style>
""" + BASE_STYLE + """
        h1 { font-size: 20px; font-weight: 500; margin-bottom: 32px; text-align: center; }
        h2 { font-size: 16px; font-weight: 400; color: #666; margin-bottom: 16px; }
        .section { margin-bottom: 32px; }
        .checkbox-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .checkbox-item { display: flex; align-items: center; gap: 10px; }
        .checkbox-item input { width: 18px; height: 18px; cursor: pointer; }
        .checkbox-item label { font-size: 14px; cursor: pointer; }
        .input-row { display: flex; align-items: center; gap: 12px; }
        input[type="number"] { 
            padding: 10px 12px; 
            border: 1px solid #d1d5db; 
            border-radius: 6px; 
            font-size: 14px; 
            width: 80px; 
        }
        input[type="number"]:focus { outline: none; border-color: #1a1a1a; }
        .hint { font-size: 13px; color: #666; }
        .btn { 
            width: 100%; 
            padding: 14px; 
            background: #1a1a1a; 
            color: #fff; 
            border: none; 
            border-radius: 6px; 
            font-size: 14px; 
            cursor: pointer; 
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .error { color: #dc3545; font-size: 13px; margin-top: 12px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Survey Item Generator</h1>
        
        <div class="section">
            <h2>Select dimensions</h2>
            <div class="checkbox-grid">
                {% for dim in dimensions %}
                <div class="checkbox-item">
                    <input type="checkbox" id="{{ dim }}" name="dimensions" value="{{ dim }}">
                    <label for="{{ dim }}">{{ dim }}</label>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2>Number of questions</h2>
            <div class="input-row">
                <input type="number" id="numQuestions" value="5" min="1" max="20">
                <span class="hint">(1-20)</span>
            </div>
        </div>
        
        <button class="btn" onclick="submit()">Generate</button>
        <div id="error" class="error" style="display:none;"></div>
    </div>
    
    <script>
        function submit() {
            const dimensions = Array.from(document.querySelectorAll('input[name="dimensions"]:checked')).map(cb => cb.value);
            const numQuestions = parseInt(document.getElementById('numQuestions').value);
            const errorDiv = document.getElementById('error');
            
            errorDiv.style.display = 'none';
            
            if (dimensions.length === 0) {
                errorDiv.textContent = 'Please select at least one dimension.';
                errorDiv.style.display = 'block';
                return;
            }
            
            if (numQuestions < 1 || numQuestions > 20) {
                errorDiv.textContent = 'Please enter a number between 1 and 20.';
                errorDiv.style.display = 'block';
                return;
            }
            
            window.location.href = '/loading?dims=' + dimensions.join(',') + '&num=' + numQuestions;
        }
    </script>
</body>
</html>
"""

LOADING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generating...</title>
    <style>
""" + BASE_STYLE + """
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #1a1a1a;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        p { text-align: center; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="spinner"></div>
        <p>Generating questions...</p>
    </div>
    
    <script>
        fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dimensions: {{ dimensions | tojson }},
                num_questions: {{ num_questions }}
            })
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                window.location.href = '/error?msg=' + encodeURIComponent(data.error);
            } else {
                window.location.href = data.redirect;
            }
        })
        .catch(err => {
            window.location.href = '/error?msg=' + encodeURIComponent('Generation failed');
        });
    </script>
</body>
</html>
"""

QUESTIONS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rate Questions</title>
    <style>
""" + BASE_STYLE + """
        h1 { font-size: 20px; font-weight: 600; margin-bottom: 24px; text-align: center; }
        .question-card { border: 1px solid #e5e5e5; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .question-text { font-size: 15px; line-height: 1.6; margin-bottom: 16px; }
        .rating-row { display: flex; justify-content: space-between; gap: 8px; }
        .rating-btn { 
            flex: 1; 
            width: 48px;
            height: 48px;
            border: 1px solid #d1d5db; 
            background: #fff; 
            border-radius: 50%; 
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .rating-btn:hover { border-color: #1a1a1a; }
        .rating-btn.selected { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
        .rating-label { font-size: 11px; color: #666; margin-top: 4px; text-align: center; }
        .rating-wrapper { flex: 1; text-align: center; }
        .btn { 
            padding: 14px 24px; 
            background: #1a1a1a; 
            color: #fff; 
            border: none; 
            border-radius: 6px; 
            font-size: 14px; 
            cursor: pointer;
            margin-top: 24px;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .error { color: #dc3545; font-size: 13px; margin-top: 12px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rate the Questions</h1>
        
        <div id="questions"></div>
        
        <button class="btn" id="submitBtn" onclick="submit()">Submit Ratings</button>
        <div id="error" class="error" style="display:none;"></div>
    </div>
    
    <script>
        const questions = {{ questions | tojson }};
        const dimensions = {{ dimensions | tojson }};
        console.log('Received questions:', questions);
        console.log('Received dimensions:', dimensions);
        const ratings = new Array(questions.length).fill(null);
        
        function displayQuestions() {
            const container = document.getElementById('questions');
            console.log('Displaying', questions.length, 'questions');
            if (!questions || questions.length === 0) {
                container.innerHTML = '<p style="padding:20px;">No questions available. Please go back and try again.</p>';
                return;
            }
            container.innerHTML = questions.map((q, idx) => `
                <div class="question-card">
                    <div class="question-text">${idx + 1}. ${q}</div>
                    <div class="rating-row">
                        ${[1,2,3,4,5].map(n => `
                            <div class="rating-wrapper">
                                <button class="rating-btn ${ratings[idx] === n ? 'selected' : ''}" onclick="selectRating(${idx}, ${n})">${n}</button>
                                <div class="rating-label">${getLabel(n)}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('');
        }
        
        function getLabel(n) {
            return ['', 'Poor', 'Fair', 'OK', 'Good', 'Great'][n];
        }
        
        function selectRating(idx, rating) {
            ratings[idx] = rating;
            displayQuestions();
        }
        
        async function submit() {
            const errorDiv = document.getElementById('error');
            errorDiv.style.display = 'none';
            
            const missing = ratings.filter(r => r === null).length;
            if (missing > 0) {
                errorDiv.textContent = 'Please rate all questions.';
                errorDiv.style.display = 'block';
                return;
            }
            
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('submitBtn').textContent = 'Saving...';
            
            const data = {
                dimensions,
                ratings: questions.map((q, i) => ({ question: q, rating: ratings[i] })),
                model_name: 'google/flan-t5-base'
            };
            
            try {
                const response = await fetch('/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.error) {
                    errorDiv.textContent = result.error;
                    errorDiv.style.display = 'block';
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = 'Submit Ratings';
                    return;
                }
                
                window.location.href = '/thanks';
            } catch (err) {
                errorDiv.textContent = 'An error occurred. Please try again.';
                errorDiv.style.display = 'block';
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('submitBtn').textContent = 'Submit Ratings';
            }
        }
        
        displayQuestions();
    </script>
</body>
</html>
"""

THANKS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thank You</title>
    <style>
""" + BASE_STYLE + """
        .content { text-align: center; }
        .icon { font-size: 48px; margin-bottom: 16px; }
        h1 { font-size: 24px; font-weight: 500; margin-bottom: 8px; }
        p { color: #666; font-size: 14px; margin-bottom: 32px; }
        .btn { 
            padding: 14px 32px; 
            background: #1a1a1a; 
            color: #fff; 
            border: none; 
            border-radius: 6px; 
            font-size: 14px; 
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="content">
            <div class="icon">&#10003;</div>
            <h1>Thank you!</h1>
            <p>Your ratings have been saved successfully.</p>
            <a href="/" class="btn">Start Over</a>
        </div>
    </div>
</body>
</html>
"""


def load_source_df() -> pd.DataFrame:
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
    return pd.read_csv(SOURCE_FILE)


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_ratings_file() -> None:
    if not os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "target_dims", "generated_text", "rating", "model_name"])


@app.route("/")
def index():
    session.clear()
    return render_template_string(WELCOME_HTML, dimensions=DIMENSIONS)


@app.route("/loading")
def loading():
    dims_str = request.args.get("dims", "")
    dimensions = [d for d in dims_str.split(",") if d]
    num_questions = int(request.args.get("num", 5))
    
    if not dimensions:
        return redirect(url_for("index"))
    
    if num_questions < 1 or num_questions > 20:
        return redirect(url_for("index"))
    
    return render_template_string(LOADING_HTML, dimensions=dimensions, num_questions=num_questions)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    import base64
    import json
    
    data = request.get_json()
    dimensions: list[str] = data.get("dimensions", [])
    num_questions: int = data.get("num_questions", 5)
    
    if not dimensions:
        return jsonify({"error": "At least one dimension must be selected."}), 400
    
    if num_questions < 1 or num_questions > 20:
        return jsonify({"error": "Number of questions must be between 1 and 20."}), 400
    
    try:
        df = load_source_df()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    
    try:
        seeds = retrieve_seeds(df, DIMENSIONS, dimensions, k=5)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    if seeds.empty:
        return jsonify({"error": "No valid seeds found for the selected dimensions."}), 400
    
    seed_texts = seed_texts_from_df(seeds)
    
    if not seed_texts:
        return jsonify({"error": "No valid seed texts found."}), 400
    
    try:
        generator = SeededQuestionGenerator()
        result = generator.collect(
            target_dims=dimensions,
            seed_texts=seed_texts,
            n_questions=num_questions,
            require_exact_count=False,
        )
    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500
    
    if not result.outputs:
        return jsonify({"error": "No valid questions could be generated."}), 400
    
    encoded = base64.urlsafe_b64encode(json.dumps({
        "questions": result.outputs,
        "dimensions": dimensions
    }, ensure_ascii=False).encode('utf-8')).decode('utf-8')
    
    return jsonify({"redirect": f"/questions?data={encoded}"})


ERROR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
    <style>
""" + BASE_STYLE + """
        .content { text-align: center; }
        h1 { font-size: 20px; font-weight: 500; margin-bottom: 16px; }
        p { color: #666; font-size: 14px; margin-bottom: 24px; }
        .btn { 
            padding: 14px 32px; 
            background: #1a1a1a; 
            color: #fff; 
            border: none; 
            border-radius: 6px; 
            font-size: 14px; 
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="content">
            <h1>Something went wrong</h1>
            <p>{{ message }}</p>
            <a href="/" class="btn">Go Back</a>
        </div>
    </div>
</body>
</html>
"""


@app.route("/questions")
def questions():
    import base64
    import json
    
    data_str = request.args.get("data", "")
    try:
        decoded = base64.urlsafe_b64decode(data_str).decode('utf-8')
        parsed = json.loads(decoded)
        question_list = parsed.get("questions", [])
        dims = parsed.get("dimensions", [])
    except Exception as e:
        print(f"Error decoding: {e}")
        return redirect(url_for("index"))
    
    if not question_list:
        return redirect(url_for("index"))
    
    return render_template_string(
        QUESTIONS_HTML,
        questions=question_list,
        dimensions=dims,
    )


@app.route("/thanks")
def thanks():
    session.clear()
    return render_template_string(THANKS_HTML)


@app.route("/error")
def error_page():
    message = request.args.get("msg", "Something went wrong.")
    return render_template_string(ERROR_HTML, message=message)


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    ratings: list[dict] = data.get("ratings", [])
    dimensions: list[str] = data.get("dimensions", [])
    model_name: Optional[str] = data.get("model_name", "")
    
    if not ratings:
        return jsonify({"error": "No ratings to save."}), 400
    
    ensure_data_dir()
    ensure_ratings_file()
    
    target_dims_str = ", ".join(dimensions)
    timestamp = datetime.now().isoformat()
    
    try:
        with open(RATINGS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for item in ratings:
                writer.writerow([
                    timestamp,
                    target_dims_str,
                    item.get("question", ""),
                    item.get("rating", ""),
                    model_name or "",
                ])
    except IOError as e:
        return jsonify({"error": f"Failed to save ratings: {str(e)}"}), 500
    
    return jsonify({"success": True, "saved": len(ratings)})


if __name__ == "__main__":
    ensure_data_dir()
    app.run(host="0.0.0.0", port=5000, debug=True)
