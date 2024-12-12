import streamlit as st
import spacy
import random
from docx import Document

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """Extract significant nouns, proper nouns, and other key terms from the text."""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2]
    return list(set(keywords))  # Remove duplicates

def generate_distractors(correct_answer, keywords):
    """Generate distractors that are semantically related to the correct answer."""
    distractors = [word for word in keywords if word.lower() != correct_answer.lower()]
    random.shuffle(distractors)
    return distractors[:3]  # Return the top 3 distractors

def generate_mcq_dynamic(text):
    """Generate MCQs with contextually relevant options and varied formats."""
    keywords = extract_keywords(text)
    doc = nlp(text)
    questions = []

    for sentence in doc.sents:
        if "is" in sentence.text or "are" in sentence.text:
            question_formats = [
                lambda ans, sent: sent.text.replace(ans, "_____"),
                lambda ans, sent: f"What is {sent.text.split('is')[1].strip()}?" if len(sent.text.split('is')) > 1 else sent.text.replace(ans, "_____"),
                lambda ans, sent: f"Which of the following is {sent.text.split('is')[0].strip().replace(ans.strip(),'')}?" if len(sent.text.split('is')) > 1 else sent.text.replace(ans, "_____")
            ]

            correct_answer = next((word for word in keywords if word in sentence.text), None)
            if correct_answer:
                distractors = generate_distractors(correct_answer, keywords)
                question_format = random.choice(question_formats)
                questions.append({
                    "question": question_format(correct_answer, sentence),
                    "correct_answer": correct_answer,
                    "options": [correct_answer] + distractors
                })
    return questions

def extract_text_from_docx(file):
    """Extracts text from a DOCX file using python-docx."""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

st.title("Dynamic MCQ Generator")

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Currently, only DOCX files are supported.")
        st.stop()

    questions = generate_mcq_dynamic(text)
    st.write(f"Generated {len(questions)} questions.")

    num_questions = st.number_input("Enter the number of questions to attempt:", min_value=1, max_value=len(questions), step=1)

    if st.button("Start Quiz"):
        selected_questions = random.sample(questions, num_questions)
        score = 0

        for i, q in enumerate(selected_questions, 1):
            st.write(f"Q{i}: {q['question']}")
            options = random.sample(q['options'], len(q['options']))  # Shuffle options

            user_answer = st.radio("Select your answer:", options, key=f"q{i}")

            if user_answer == q['correct_answer']:
                score += 1
                st.write("Correct!")
            else:
                st.write(f"Incorrect. The correct answer is: {q['correct_answer']}")

        st.write(f"You scored {score}/{num_questions}")
