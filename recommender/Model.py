import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import re
import hashlib
from graphviz import Digraph
from dotenv import load_dotenv
import os

# === Load API Key from .env ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

# === LLaMA API Call using OpenRouter ===
def call_llama_api(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Career-Path-Recommender"
    }
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"\n❌ LLaMA API Error: {e}")
        return None

# === Main callable for backend ===
def generate_recommendation_from_input(user_interest, user_qualification):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "final_course_data_for_bert.csv")
    df = pd.read_csv(csv_path)
    df['course_summary'] = "Course: " + df['Course'] + " | Field: " + df['Field']
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df_filtered = df[df['Minimum_Qualification'].str.lower() == user_qualification.lower()]
    if df_filtered.empty:
        raise ValueError("No courses found for your qualification.")

    course_texts = df_filtered['course_summary'].tolist()
    course_embeddings = model.encode(course_texts, show_progress_bar=False)
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
    related_response = call_llama_api(prompt_related)

    try:
        related_courses = json.loads(related_response)
    except:
        related_courses = []
        for line in related_response.strip().split('\n'):
            match = re.match(r'^\d+\. (.+)', line.strip())
            if match:
                related_courses.append(match.group(1).strip())

    all_courses = [best_course] + related_courses
    selected_course = best_course  # Auto-select top match for roadmap

    # === Roadmap prompt ===
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
    roadmap_json = call_llama_api(prompt_roadmap)
    if not roadmap_json:
        raise ValueError("Failed to generate roadmap.")

    try:
        json_data = json.loads(roadmap_json)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', roadmap_json)
        if match:
            cleaned_json = match.group(0)
            json_data = json.loads(cleaned_json)
        else:
            raise ValueError("Invalid roadmap JSON.")

    return {
        "recommended_course": best_course,
        "related_courses": all_courses,
        "roadmap": json_data.get("roadmap", {})
    }

# === CLI Testing ===
if __name__ == "__main__":
    interest = input("🎤 Enter your interests: ")
    qualification = input("🎓 Enter your current qualification: ").strip().lower()
    result = generate_recommendation_from_input(interest, qualification)
    print(json.dumps(result, indent=2))
