from flask import Flask, render_template, request, redirect, url_for
from utils import extract_text_from_pdf, extract_text_from_docx, generate_mcq_dynamic

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    file = request.files["file"]
    if not file:
        return "No file uploaded!", 400

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        return "Unsupported file type!", 400

    questions = generate_mcq_dynamic(text)
    return render_template("quiz.html", questions=questions)

@app.route("/submit", methods=["POST"])
def submit():
    user_answers = request.form
    results = []
    score = 0

    for i, q in enumerate(questions):
        user_answer = user_answers.get(f"q{i+1}")
        correct_answer = q["correct_answer"]
        is_correct = user_answer == correct_answer
        if is_correct:
            score += 1
        results.append({
            "question": q["question"],
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "is_correct": is_correct,
        })

    return render_template("result.html", score=score, total=len(questions), results=results)

if __name__ == "__main__":
    app.run(debug=True)