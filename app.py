from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from generator import (
    GenerationResult,
    SeededQuestionGenerator,
    retrieve_seeds,
    seed_texts_from_df,
)

app = Flask(__name__)

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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Item Generator</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .checkbox-group { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .checkbox-item { display: flex; align-items: center; gap: 8px; }
        input[type="number"] { padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; width: 100px; }
        button { background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #545b62; }
        .question-card { background: white; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin-bottom: 15px; }
        .question-text { font-size: 18px; margin-bottom: 15px; color: #333; }
        .rating-group { display: flex; gap: 10px; }
        .rating-btn { padding: 8px 16px; border: 1px solid #ddd; background: white; border-radius: 4px; cursor: pointer; }
        .rating-btn:hover { background: #e9ecef; }
        .rating-btn.selected { background: #007bff; color: white; border-color: #007bff; }
        .rating-label { font-size: 12px; color: #666; margin-top: 5px; text-align: center; }
        .rating-wrapper { display: flex; flex-direction: column; align-items: center; }
        .error { color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .success { color: #155724; background: #d4edda; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .loading { color: #666; font-style: italic; }
        input[type="checkbox"] { width: 18px; height: 18px; }
        label { cursor: pointer; }
    </style>
</head>
<body>
    <h1>Survey Item Generator</h1>
    
    <div class="section">
        <h2>1. Select Dimensions</h2>
        <p>Select one or more target dimensions:</p>
        <div class="checkbox-group">
            {% for dim in dimensions %}
            <div class="checkbox-item">
                <input type="checkbox" id="{{ dim }}" name="dimensions" value="{{ dim }}">
                <label for="{{ dim }}">{{ dim }}</label>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="section">
        <h2>2. Number of Questions</h2>
        <input type="number" id="numQuestions" value="5" min="1" max="20">
        <small style="color: #666;">(1-20 questions)</small>
    </div>
    
    <button id="generateBtn" onclick="generateQuestions()">Generate Questions</button>
    
    <div id="message"></div>
    <div id="loading" class="loading" style="display:none;">Generating questions, please wait...</div>
    
    <div id="questionsContainer" style="margin-top: 30px;"></div>
    
    <div id="saveSection" style="display:none; margin-top: 20px;">
        <button onclick="saveRatings()">Save Ratings</button>
    </div>

    <script>
        let currentQuestions = [];
        
        async function generateQuestions() {
            const dimensions = Array.from(document.querySelectorAll('input[name="dimensions"]:checked')).map(cb => cb.value);
            const numQuestions = parseInt(document.getElementById('numQuestions').value);
            const messageDiv = document.getElementById('message');
            const loadingDiv = document.getElementById('loading');
            const questionsDiv = document.getElementById('questionsContainer');
            const saveSection = document.getElementById('saveSection');
            
            messageDiv.innerHTML = '';
            questionsDiv.innerHTML = '';
            saveSection.style.display = 'none';
            
            if (dimensions.length === 0) {
                messageDiv.innerHTML = '<div class="error">Please select at least one dimension.</div>';
                return;
            }
            
            if (numQuestions < 1 || numQuestions > 20) {
                messageDiv.innerHTML = '<div class="error">Please enter a number between 1 and 20.</div>';
                return;
            }
            
            loadingDiv.style.display = 'block';
            document.getElementById('generateBtn').disabled = true;
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dimensions, num_questions: numQuestions })
                });
                
                const data = await response.json();
                loadingDiv.style.display = 'none';
                document.getElementById('generateBtn').disabled = false;
                
                if (data.error) {
                    messageDiv.innerHTML = '<div class="error">' + data.error + '</div>';
                    return;
                }
                
                currentQuestions = data.questions;
                displayQuestions(currentQuestions);
                saveSection.style.display = 'block';
            } catch (err) {
                loadingDiv.style.display = 'none';
                document.getElementById('generateBtn').disabled = false;
                messageDiv.innerHTML = '<div class="error">An error occurred. Please try again.</div>';
            }
        }
        
        function displayQuestions(questions) {
            const questionsDiv = document.getElementById('questionsContainer');
            questionsDiv.innerHTML = '';
            
            questions.forEach((q, idx) => {
                const card = document.createElement('div');
                card.className = 'question-card';
                card.innerHTML = `
                    <div class="question-text">${idx + 1}. ${q}</div>
                    <div class="rating-group" id="rating-${idx}">
                        ${[1,2,3,4,5].map(n => `
                            <div class="rating-wrapper">
                                <button class="rating-btn" onclick="selectRating(${idx}, ${n})">${n}</button>
                                <div class="rating-label">${getRatingLabel(n)}</div>
                            </div>
                        `).join('')}
                    </div>
                    <input type="hidden" id="hidden-rating-${idx}" value="">
                `;
                questionsDiv.appendChild(card);
            });
        }
        
        function getRatingLabel(n) {
            const labels = {1: 'Very Poor', 2: 'Poor', 3: 'Acceptable', 4: 'Good', 5: 'Very Good'};
            return labels[n];
        }
        
        function selectRating(questionIdx, rating) {
            const group = document.getElementById('rating-' + questionIdx);
            group.querySelectorAll('.rating-btn').forEach((btn, i) => {
                btn.classList.toggle('selected', i + 1 === rating);
            });
            document.getElementById('hidden-rating-' + questionIdx).value = rating;
        }
        
        async function saveRatings() {
            const ratings = [];
            const messageDiv = document.getElementById('message');
            
            for (let i = 0; i < currentQuestions.length; i++) {
                const rating = document.getElementById('hidden-rating-' + i).value;
                if (!rating) {
                    messageDiv.innerHTML = '<div class="error">Please rate all questions before saving.</div>';
                    return;
                }
                ratings.push({
                    question: currentQuestions[i],
                    rating: parseInt(rating)
                });
            }
            
            const dimensions = Array.from(document.querySelectorAll('input[name="dimensions"]:checked')).map(cb => cb.value);
            
            try {
                const response = await fetch('/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        dimensions,
                        ratings,
                        model_name: 'google/flan-t5-base'
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    messageDiv.innerHTML = '<div class="error">' + data.error + '</div>';
                    return;
                }
                
                messageDiv.innerHTML = '<div class="success">Ratings saved successfully!</div>';
                document.getElementById('questionsContainer').innerHTML = '';
                document.getElementById('saveSection').style.display = 'none';
                currentQuestions = [];
            } catch (err) {
                messageDiv.innerHTML = '<div class="error">An error occurred while saving. Please try again.</div>';
            }
        }
    </script>
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
            writer.writerow(["timestamp", "target_dims", "generated_text", "rating", "model_name", "prompt"])


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, dimensions=DIMENSIONS)


@app.route("/generate", methods=["POST"])
def generate():
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

    return jsonify({
        "questions": result.outputs,
        "model_name": result.prompt[:100] if result.prompt else "",
        "prompt": result.prompt,
    })


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    ratings: list[dict] = data.get("ratings", [])
    dimensions: list[str] = data.get("dimensions", [])
    model_name: Optional[str] = data.get("model_name", "")
    prompt: Optional[str] = data.get("prompt", "")

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
                    prompt or "",
                ])
    except IOError as e:
        return jsonify({"error": f"Failed to save ratings: {str(e)}"}), 500

    return jsonify({"success": True, "saved": len(ratings)})


if __name__ == "__main__":
    ensure_data_dir()
    app.run(host="0.0.0.0", port=5000, debug=True)
