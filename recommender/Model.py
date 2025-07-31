import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import hashlib
from graphviz import Digraph

# === Load .env and configure Gemini ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

# === Utility Functions ===
def sanitize_filename(title):
    return re.sub(r'[^\w\-]', '', title.strip())

def hash_id(value):
    return hashlib.md5(value.encode()).hexdigest()[:8]

def add_node(dot, node_id, label, shape='box', style='rounded,filled', fillcolor='lightblue'):
    dot.node(node_id, label, shape=shape, style=style, fillcolor=fillcolor)

def build_flowchart(dot, data, parent_id=None, level=0):
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lavender', 'peachpuff', 'mistyrose', 'honeydew', 'thistle']
    current_color = colors[level % len(colors)]

    if isinstance(data, dict):
        for key, value in data.items():
            node_label = str(key).strip()
            key_id = hash_id(f"{parent_id}_{node_label}" if parent_id else node_label)
            add_node(dot, key_id, node_label, fillcolor=current_color)
            if parent_id:
                dot.edge(parent_id, key_id)
            build_flowchart(dot, value, key_id, level + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                build_flowchart(dot, item, parent_id, level + 1)
            else:
                item_label = str(item).strip()
                item_id = hash_id(f"{parent_id}_{item_label}")
                add_node(dot, item_id, f"• {item_label}", shape='box', style='filled', fillcolor='white')
                if parent_id:
                    dot.edge(parent_id, item_id)
    else:
        if parent_id and str(data).strip():
            val_label = str(data).strip()
            val_id = hash_id(f"{parent_id}_{val_label}")
            add_node(dot, val_id, val_label, shape='ellipse', fillcolor='white')
            dot.edge(parent_id, val_id)

def call_gemini_api(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"\n❌ Gemini API Error: {e}")
        return None

# === Load Dataset and Model at Top-Level Once ===
df = pd.read_csv("final_course_data_for_bert.csv")
df['course_summary'] = "Course: " + df['Course'] + " | Field: " + df['Field']
model = SentenceTransformer('all-MiniLM-L6-v2')

# === MAIN FUNCTION to be called from views.py ===
def generate_recommendation_from_input(user_interest, user_qualification):
    user_qualification = user_qualification.strip().lower()
    df_filtered = df[df['Minimum_Qualification'].str.lower() == user_qualification]

    if df_filtered.empty:
        raise ValueError("No courses found for the given qualification.")

    course_texts = df_filtered['course_summary'].tolist()
    course_embeddings = model.encode(course_texts)
    user_embedding = model.encode([user_interest])
    similarities = cosine_similarity(user_embedding, course_embeddings)[0]
    best_index = similarities.argmax()
    best_row = df_filtered.iloc[best_index]
    best_course = best_row['Course']

    course_prefix = best_course.split()[0].upper()
    prompt_related = (
        f"Suggest 4 other full {course_prefix} engineering degree courses that are closely related to '{best_course}'. "
        "Return only a numbered list without explanation. Format strictly as raw JSON array."
    )
    related_courses_response = call_gemini_api(prompt_related)

    if not related_courses_response:
        print("⚠ Using dummy related courses due to Gemini quota.")
    related_courses = [
        "Bachelor of Engineering in Cybersecurity",
        "B.Tech in Information Security",
        "Bachelor of Science in Digital Forensics",
        "Cyber Law and Security Management"
    ]


    try:
        related_courses = json.loads(related_courses_response)
        if not isinstance(related_courses, list):
            raise ValueError("Gemini did not return a list.")
    except Exception as e:
        raise ValueError(f"Error parsing related courses from Gemini: {e}")

    all_courses = [best_course] + related_courses
    selected_course = all_courses[0]  # Default to top recommended

    # Gemini Prompt for Roadmap
    prompt_roadmap = f"""
    You are an expert mentor and a caring parent helping your child succeed in the course '{selected_course}'.

    Create a deeply structured, spoon-feeding style 4-year roadmap from scratch to expert level.

    Organize everything in proper hierarchy as valid JSON.

    Include:
    1. Semester-wise academic curriculum (subjects, foundational to advanced)
    2. Skills to build in each phase (technical + soft skills)
    3. Online course recommendations (free + paid; mention platforms like Coursera, Udemy, edX, etc.)
    4. Weekly or monthly learning milestones
    5. Mini and major project suggestions with themes and tech stacks
    6. Portfolio-building ideas and GitHub tips
    7. Personality development and communication improvement activities
    8. Events to participate in (Hackathons, meetups, competitions, open source)
    9. Internship search strategy, resume and LinkedIn optimization, interview prep
    10. Final-year job placement guide (roles to target, top companies, mock interviews)

    Format everything as a JSON with the root key "roadmap" and include the "title" as the course name. Each item should be labeled clearly.
    Use descriptive labels for nodes. Use structured nesting.
    Do NOT explain anything outside JSON.
    Output only valid JSON.
    """

    roadmap_json = call_gemini_api(prompt_roadmap)
    if not roadmap_json:
        raise ValueError("Failed to generate academic roadmap from Gemini.")

    try:
        json_data = json.loads(roadmap_json)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', roadmap_json)
        if match:
            cleaned_json = match.group(0)
            json_data = json.loads(cleaned_json)
        else:
            raise ValueError("No valid JSON block found.")

    with open("flowchart.json", "w") as f:
        json.dump(json_data, f, indent=2)

    title = json_data.get("roadmap", {}).get("title", "Career_Roadmap")
    dot = Digraph(comment=title,
                  graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'size': '20,30', 'ratio': 'compress'},
                  node_attr={'fontname': 'Helvetica', 'fontsize': '12', 'width': '1.5', 'height': '0.8'},
                  edge_attr={'fontsize': '10'})

    title_id = hash_id('title')
    dot.node(title_id, title, shape='doubleoctagon', style='filled', fillcolor='dodgerblue',
             fontcolor='white', fontsize='18', width='2.0', height='1.0')

    build_flowchart(dot, json_data, title_id)
    pdf_path = "courses.pdf"
    dot.render("courses", format="pdf", cleanup=True)

    return {
        "course": best_course,
        "related_courses": all_courses,
        "selected": selected_course,
        "pdf_path": pdf_path,
    }
