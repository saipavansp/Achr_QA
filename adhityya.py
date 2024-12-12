import streamlit as st
import spacy
import random
import PyPDF2
import docx
import re
from io import BytesIO



class MCQGenerator:
    def __init__(self):
        # Load spaCy model
        @st.cache_resource
        def load_model():
            return spacy.load("en_core_web_sm")

        self.nlp = load_model()
        self.nlp.add_pipe('sentencizer')

    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded PDF or DOCX file"""
        try:
            # Get the file bytes
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

    def extract_keywords(self, doc):
        """Extract significant keywords from the document"""
        # Get nouns and proper nouns
        basic_keywords = {token.text for token in doc
                          if token.pos_ in ["NOUN", "PROPN"]
                          and len(token.text) > 2}

        # Get important verbs and adjectives
        additional_keywords = {token.text for token in doc
                               if token.pos_ in ["VERB", "ADJ"]
                               and token.dep_ in ["ROOT", "acomp", "attr"]
                               and len(token.text) > 2}

        return basic_keywords.union(additional_keywords)

    def generate_distractors(self, correct_answer, keywords, context, q_type):
        """Generate contextually relevant distractors"""
        candidates = []

        # Basic candidates from keywords
        basic_candidates = [word for word in keywords
                            if word.lower() != correct_answer.lower()
                            and len(word) > 2]

        # Type-specific distractor generation
        if q_type == "NUMERICAL":
            try:
                num = float(correct_answer)
                variations = [num * 0.8, num * 1.2, num * 0.5]
                candidates.extend([str(int(v)) if v.is_integer() else f"{v:.1f}"
                                   for v in variations])
            except ValueError:
                candidates.extend(basic_candidates)

        elif q_type == "DEFINITION":
            doc = self.nlp(context)
            candidates.extend([token.text for token in doc
                               if token.pos_ in ["NOUN", "PROPN"]
                               and token.text.lower() != correct_answer.lower()])

        else:
            candidates.extend(basic_candidates)

        candidates = list(set(candidates))
        if len(candidates) < 3:
            candidates.extend([word for word in keywords
                               if word not in candidates
                               and word.lower() != correct_answer.lower()])

        return random.sample(candidates, min(3, len(candidates))) if candidates else []

    def create_question(self, sentence, keywords, questions, q_type):
        """Create a question based on sentence type"""
        possible_answers = keywords.intersection(set(sentence.split()))

        if not possible_answers:
            return

        if q_type == "NUMERICAL":
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
            if numbers:
                answer = random.choice(numbers)
            else:
                answer = random.choice(list(possible_answers))
        else:
            answer = random.choice(list(possible_answers))

        distractors = self.generate_distractors(answer, keywords, sentence, q_type)

        if len(distractors) >= 3:
            question_text = sentence.replace(answer, "_____")

            if q_type == "DEFINITION":
                if "is defined as" in sentence:
                    question_text = f"What {sentence.split('is defined as')[0].strip()} is defined as?"
            elif q_type == "CAUSE_EFFECT":
                if "because" in sentence:
                    question_text = f"What is the effect of {answer} in this context?"

            questions.append({
                "question": question_text,
                "correct_answer": answer,
                "options": [answer] + distractors,
                "type": q_type
            })

    def generate_mcq_dynamic(self, text):
        """Generate MCQs from text with enhanced sentence analysis"""
        chunk_size = 5000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        questions = []
        keywords = set()

        for chunk in chunks:
            doc = self.nlp(chunk)
            chunk_keywords = self.extract_keywords(doc)
            keywords.update(chunk_keywords)

            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text.split()) < 4:
                    continue

                # Check various sentence types
                if any(verb in sent_text.lower() for verb in
                       ["is", "are", "has", "have", "was", "were", "contains", "includes"]):
                    self.create_question(sent_text, keywords, questions, "FACT")

                elif any(conj in sent_text.lower() for conj in ["because", "since", "therefore", "as a result"]):
                    self.create_question(sent_text, keywords, questions, "CAUSE_EFFECT")

                elif any(definer in sent_text.lower() for definer in ["is defined as", "means", "refers to"]):
                    self.create_question(sent_text, keywords, questions, "DEFINITION")

                elif any(comp in sent_text.lower() for comp in
                         ["more than", "less than", "as much as", "the largest", "the smallest"]):
                    self.create_question(sent_text, keywords, questions, "COMPARISON")

                elif any(word in sent_text.lower() for word in ["if", "when", "unless"]):
                    self.create_question(sent_text, keywords, questions, "CONDITIONAL")

                elif any(num in sent_text for num in "0123456789"):
                    self.create_question(sent_text, keywords, questions, "NUMERICAL")

                if len(questions) >= 50:
                    return questions

        return questions


# [Previous imports and class methods remain the same until the main() function]

def main():
    st.title("Enhanced MCQ Generator")
    st.write("Upload a PDF or DOCX file to generate multiple-choice questions.")

    # Initialize session state if not exists
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'submitted_answers' not in st.session_state:
        st.session_state.submitted_answers = {}
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False

    def next_question():
        st.session_state.submitted_answers[st.session_state.current_question] = st.session_state.get(
            f"q_{st.session_state.current_question}")
        st.session_state.current_question += 1

    def prev_question():
        st.session_state.current_question -= 1

    def submit_quiz():
        st.session_state.submitted_answers[st.session_state.current_question] = st.session_state.get(
            f"q_{st.session_state.current_question}")
        st.session_state.quiz_completed = True
        # Calculate score
        st.session_state.score = 0
        for i, q in enumerate(st.session_state.questions):
            if i in st.session_state.submitted_answers:
                if st.session_state.submitted_answers[i] == q['correct_answer']:
                    st.session_state.score += 1

    def start_new_quiz():
        st.session_state.questions = []
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.submitted_answers = {}
        st.session_state.quiz_completed = False

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'])

    if uploaded_file is not None:
        # Number of questions selector
        num_questions = st.number_input("Number of questions", min_value=1, max_value=50, value=5)

        # Generate button
        if st.button("Generate Quiz"):
            with st.spinner("Generating questions..."):
                # Create MCQ generator instance
                mcq_gen = MCQGenerator()

                # Extract text from file
                text = mcq_gen.extract_text_from_file(uploaded_file)

                if text:
                    # Generate questions
                    all_questions = mcq_gen.generate_mcq_dynamic(text)
                    if all_questions:
                        st.session_state.questions = random.sample(all_questions,
                                                                   min(num_questions, len(all_questions)))
                        st.session_state.current_question = 0
                        st.session_state.score = 0
                        st.session_state.submitted_answers = {}
                        st.session_state.quiz_completed = False
                        st.success(f"Generated {len(st.session_state.questions)} questions!")
                    else:
                        st.error("No questions could be generated from the text.")

        # Display quiz if questions are generated
        if st.session_state.questions:
            if not st.session_state.quiz_completed:
                # Display current question
                current_q = st.session_state.questions[st.session_state.current_question]

                # Create columns for question number and type
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"Question {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
                with col2:
                    st.write(f"Type: {current_q['type']}")

                st.write(current_q['question'])

                # Radio buttons for options
                st.radio(
                    "Select your answer:",
                    current_q['options'],
                    key=f"q_{st.session_state.current_question}"
                )

                # Navigation buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.session_state.current_question > 0:
                        st.button("Previous", on_click=prev_question)

                with col2:
                    if st.session_state.current_question < len(st.session_state.questions) - 1:
                        st.button("Next", on_click=next_question)

                with col3:
                    st.button("Submit Quiz", on_click=submit_quiz)

            else:
                # Display results
                st.title("Quiz Results")
                st.write(f"Your score: {st.session_state.score}/{len(st.session_state.questions)}")
                st.write(f"Percentage: {(st.session_state.score / len(st.session_state.questions)) * 100:.2f}%")

                # Display detailed results
                st.subheader("Detailed Results")
                for i, q in enumerate(st.session_state.questions):
                    with st.expander(f"Question {i + 1} ({q['type']})"):
                        st.write(q['question'])
                        st.write(f"Your answer: {st.session_state.submitted_answers.get(i, 'Not answered')}")
                        st.write(f"Correct answer: {q['correct_answer']}")
                        if st.session_state.submitted_answers.get(i) == q['correct_answer']:
                            st.success("Correct! ✅")
                        else:
                            st.error("Incorrect ❌")

                st.button("Start New Quiz", on_click=start_new_quiz)


if __name__ == "__main__":
    main()