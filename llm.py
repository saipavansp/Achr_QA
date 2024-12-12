import streamlit as st
import spacy
import random
import PyPDF2
import docx
import re
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional
import nltk


class EnhancedMCQGenerator:
    def __init__(self):
        # Load spaCy model
        @st.cache_resource
        def load_spacy_model():
            return spacy.load("en_core_web_sm")

        @st.cache_resource
        def load_llm_model():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model = AutoModelForCausalLM.from_pretrained(
                'fbellame/pdf_to_quizz_llama_13B',
                device_map={"": device},
                load_in_4bit=True
            )
            tokenizer = AutoTokenizer.from_pretrained("fbellame/pdf_to_quizz_llama_13B", use_fast=False)
            return model, tokenizer

        self.nlp = load_spacy_model()
        self.nlp.add_pipe('sentencizer')
        self.model, self.tokenizer = load_llm_model()
        nltk.download('punkt')

    def extract_text(self, uploaded_file) -> Optional[str]:
        """Extract text from uploaded file"""
        try:
            file_bytes = uploaded_file.read()

            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            elif uploaded_file.name.endswith('.docx'):
                doc = docx.Document(BytesIO(file_bytes))
                return ' '.join(paragraph.text for paragraph in doc.paragraphs)
            else:
                st.error("Unsupported file format")
                return None
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    def generate_llm_question(self, context: str) -> Optional[Dict[str, Any]]:
        """Generate question using LLM"""
        try:
            prompt = f"""<|prompt|>You are a teacher preparing questions for a quiz. Given the following document, please generate 1 multiple-choice question (MCQ) with 4 options and a corresponding answer letter based on the document.
            Example question:
            Question: question here
            CHOICE_A: choice here
            CHOICE_B: choice here
            CHOICE_C: choice here
            CHOICE_D: choice here
            Answer: A or B or C or D
            <Begin Document>
            {context}
            <End Document></s><|answer|>"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=500,
                temperature=0.1,
                top_p=0.15,
                repetition_penalty=1.2
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse response using regex
            question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", response)
            options_match = re.findall(r"CHOICE_([A-D]):\s*(.*?)(?:\n|$)", response)
            answer_match = re.search(r"Answer:\s*([A-D])", response)

            if not all([question_match, options_match, answer_match]):
                return None

            return {
                "question": question_match.group(1),
                "options": {opt[0]: opt[1].strip() for opt in options_match},
                "correct_answer": answer_match.group(1),
                "type": "LLM_GENERATED"
            }
        except Exception as e:
            st.error(f"Error generating LLM question: {str(e)}")
            return None

    def generate_spacy_question(self, sentence: str, keywords: set) -> Optional[Dict[str, Any]]:
        """Generate question using Spacy analysis"""
        try:
            doc = self.nlp(sentence)

            # Extract potential answer candidates
            candidates = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "NUM"]]
            if not candidates:
                return None

            answer = random.choice(candidates)
            question_text = sentence.replace(answer, "_____")

            # Generate distractors
            distractors = [word for word in keywords
                           if word != answer and len(word) > 2][:3]

            if len(distractors) < 3:
                return None

            options = ['A', 'B', 'C', 'D']
            correct_index = random.randint(0, 3)
            all_options = {}

            for i, opt in enumerate(options):
                if i == correct_index:
                    all_options[opt] = answer
                else:
                    if distractors:
                        all_options[opt] = distractors.pop()
                    else:
                        return None

            return {
                "question": question_text,
                "options": all_options,
                "correct_answer": options[correct_index],
                "type": "SPACY_GENERATED"
            }
        except Exception as e:
            st.error(f"Error generating Spacy question: {str(e)}")
            return None

    def generate_questions(self, text: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate questions using both LLM and Spacy"""
        questions = []
        try:
            doc = self.nlp(text)
            keywords = {token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "NUM"]}

            # Split text into chunks
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]

            # Generate questions using both methods
            for sentence in sentences:
                if len(questions) >= num_questions:
                    break

                # Alternate between LLM and Spacy
                if len(questions) % 2 == 0:
                    question = self.generate_llm_question(sentence)
                else:
                    question = self.generate_spacy_question(sentence, keywords)

                if question is not None:
                    questions.append(question)

            return questions[:num_questions]
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}")
            return []


def main():
    st.title("Enhanced MCQ Generator")
    st.write("Upload a PDF or DOCX file to generate multiple-choice questions.")

    # Initialize session state
    for key in ['questions', 'current_question', 'score', 'submitted_answers', 'quiz_completed']:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'questions' else 0 if key in ['current_question',
                                                                               'score'] else {} if key == 'submitted_answers' else False

    # Navigation functions
    def next_question():
        st.session_state.submitted_answers[st.session_state.current_question] = st.session_state[
            f"q_{st.session_state.current_question}"]
        st.session_state.current_question += 1

    def prev_question():
        st.session_state.current_question -= 1

    def submit_quiz():
        st.session_state.submitted_answers[st.session_state.current_question] = st.session_state[
            f"q_{st.session_state.current_question}"]
        st.session_state.quiz_completed = True
        st.session_state.score = sum(1 for i, q in enumerate(st.session_state.questions)
                                     if st.session_state.submitted_answers.get(i) == q['correct_answer'])

    def start_new_quiz():
        for key in ['questions', 'current_question', 'score', 'submitted_answers', 'quiz_completed']:
            st.session_state[key] = [] if key == 'questions' else 0 if key in ['current_question',
                                                                               'score'] else {} if key == 'submitted_answers' else False

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'])

    if uploaded_file is not None:
        num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=5)

        if st.button("Generate Quiz"):
            with st.spinner("Generating questions..."):
                mcq_gen = EnhancedMCQGenerator()
                text = mcq_gen.extract_text(uploaded_file)

                if text:
                    questions = mcq_gen.generate_questions(text, num_questions)
                    if questions:
                        st.session_state.questions = questions
                        start_new_quiz()
                        st.success(f"Generated {len(questions)} questions!")
                    else:
                        st.error("No questions could be generated from the text.")

        # Display quiz
        if st.session_state.questions and not st.session_state.quiz_completed:
            current_q = st.session_state.questions[st.session_state.current_question]

            # Question header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Question {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
            with col2:
                st.write(f"Type: {current_q['type']}")

            # Question and options
            st.write(current_q['question'])
            options = [(k, v) for k, v in current_q['options'].items()]
            st.radio("Select your answer:",
                     options,
                     format_func=lambda x: f"{x[0]}: {x[1]}",
                     key=f"q_{st.session_state.current_question}")

            # Navigation
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.session_state.current_question > 0:
                    st.button("Previous", on_click=prev_question)
            with col2:
                if st.session_state.current_question < len(st.session_state.questions) - 1:
                    st.button("Next", on_click=next_question)
            with col3:
                st.button("Submit Quiz", on_click=submit_quiz)

        # Display results
        elif st.session_state.quiz_completed:
            st.title("Quiz Results")
            percentage = (st.session_state.score / len(st.session_state.questions)) * 100
            st.write(f"Your score: {st.session_state.score}/{len(st.session_state.questions)} ({percentage:.2f}%)")

            st.subheader("Detailed Results")
            for i, q in enumerate(st.session_state.questions):
                with st.expander(f"Question {i + 1} ({q['type']})"):
                    st.write(q['question'])
                    user_answer = st.session_state.submitted_answers.get(i, 'Not answered')
                    st.write(f"Your answer: {user_answer}")
                    st.write(f"Correct answer: {q['correct_answer']}")
                    if user_answer == q['correct_answer']:
                        st.success("Correct! ✅")
                    else:
                        st.error("Incorrect ❌")

            st.button("Start New Quiz", on_click=start_new_quiz)


if __name__ == "__main__":
    main()