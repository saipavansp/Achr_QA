import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import wordnet as wn
from pke.unsupervised import TopicRank
import pke
import nltk
from PyPDF2 import PdfReader
import os
import random

nltk.download('wordnet')
nltk.download('omw-1.4')

# Ensure SentencePiece library is installed
try:
    import sentencepiece
except ImportError:
    os.system('pip install sentencepiece')

# Load models and tokenizers
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    inputs = bart_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_keywords(text):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text, language="en")
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=5)
    return [phrase for phrase, _ in keyphrases]

def generate_distractors(keyword):
    distractors = []
    synsets = wn.synsets(keyword)
    if synsets:
        for hypernym in synsets[0].hypernyms():
            for lemma in hypernym.lemmas():
                distractor = lemma.name().replace('_', ' ')
                if distractor.lower() != keyword.lower():
                    distractors.append(distractor)
    return distractors[:3]

def generate_question(text, keyword):
    input_text = f"generate question: {text}"
    inputs = t5_tokenizer(input_text, return_tensors="pt")
    output = t5_model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    question = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return question

def conduct_quiz(mcqs):
    st.header("Quiz Time!")
    correct_answers = 0
    total_questions = len(mcqs)

    for i, mcq in enumerate(mcqs):
        st.subheader(f"Question {i + 1}: {mcq['question']}")
        options = [mcq["correct_answer"]] + mcq["distractors"]
        random.shuffle(options)
        selected_option = st.radio("Choose your answer:", options, key=f"q{i}")

        if st.button(f"Submit Answer for Question {i + 1}", key=f"submit{i}"):
            if selected_option == mcq["correct_answer"]:
                st.success("Correct!")
                correct_answers += 1
            else:
                st.error(f"Wrong! The correct answer was: {mcq['correct_answer']}")

    if st.button("Show Final Results"):
        st.subheader("Quiz Results")
        st.write(f"You got {correct_answers} out of {total_questions} correct.")

def main():
    st.title("MCQ Quiz Generator")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Step 1: Extract Text
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        # Step 2: Summarize Text
        with st.spinner("Summarizing text..."):
            summarized_text = summarize_text(text)
            st.subheader("Summarized Text:")
            st.write(summarized_text)

        # Step 3: Extract Keywords
        with st.spinner("Extracting keywords..."):
            keywords = extract_keywords(summarized_text)
            st.subheader("Extracted Keywords:")
            st.write(keywords)

        # Step 4: Generate MCQs
        mcqs = []
        with st.spinner("Generating MCQs..."):
            for keyword in keywords:
                question = generate_question(summarized_text, keyword)
                distractors = generate_distractors(keyword)
                if len(distractors) >= 3:
                    mcqs.append({
                        "question": question,
                        "correct_answer": keyword,
                        "distractors": distractors
                    })

        if mcqs:
            if st.button("Generate Quiz"):
                conduct_quiz(mcqs)

if __name__ == "__main__":
    main()
