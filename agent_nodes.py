# agent_nodes.py
# Contains all the functions that act as nodes in our LangGraph agent.

import re
import json
from typing import TypedDict, Optional, List

# Import shared resources from config.py
from config import df, text_generation_pipeline, tokenizer, MOVIE_GENRES

# --- Agent State Definition ---
class AgentState(TypedDict):
    question: str
    request_type: Optional[str]
    extracted_title: Optional[str]
    extracted_genres: Optional[List[str]]
    extracted_keywords: Optional[List[str]]
    validated_title_id: Optional[int]
    candidate_list: Optional[List[dict]]
    analysis_result: Optional[str]
    recommendation: Optional[str]


# --- Node Functions ---

def extract_intent_node(state: AgentState):
    """Step 1: Use LLM to extract title, genres, AND keywords."""
    print("🧐 Node: extract_intent_node")
    question = state["question"]
    genres_str = ", ".join(sorted(list(MOVIE_GENRES)))
    
    prompt_content = f"""You are an expert at analyzing user requests for movie recommendations.
Your task is to parse the user's query to identify a potential movie title, relevant genres, and specific keywords.

**Available GENRES:**
{genres_str}

**User's Query:** "{question}"

**Instructions:**
1. Identify a specific movie **title** if mentioned. If none, use null.
2. Identify relevant **genres** from the provided list.
3. Identify specific **keywords** or themes from the query (e.g., "time travel", "zombie").
4. Respond with ONLY a single JSON object.

**JSON Output Format:**
{{
  "title": "A Movie Title | null",
  "genres": ["Genre1", "Genre2"],
  "keywords": ["keyword1", "keyword2"]
}}

Your JSON Output:
"""
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True , enable_thinking=False)
    response = text_generation_pipeline(formatted_prompt)[0]['generated_text']
    
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        print("  - ERROR: LLM did not return a valid JSON object.")
        return {"extracted_title": None, "extracted_genres": [], "extracted_keywords": []}
        
    try:
        data = json.loads(match.group(0))
        extracted_title = data.get("title")
        if isinstance(extracted_title, str) and extracted_title.lower() == 'null':
            print("  - LLM returned 'null' string, converting to None.")
            extracted_title = None
            
        return {
            "extracted_title": extracted_title,
            "extracted_genres": data.get("genres", []),
            "extracted_keywords": data.get("keywords", [])
        }
    except json.JSONDecodeError:
        print("  - ERROR: Failed to decode JSON from LLM response.")
        return {"extracted_title": None, "extracted_genres": [], "extracted_keywords": []}


def verify_title_and_plan_node(state: AgentState):
    """Verifies the extracted title and plans the search strategy."""
    print("🚦 Node: verify_title_and_plan_node")
    potential_title = state.get("extracted_title")

    if not potential_title:
        print("  - No potential title found. Routing to CATEGORY search.")
        return {"request_type": "CATEGORY", "validated_title_id": None}

    print(f"  - Verifying title: '{potential_title}'...")
    
    matches = df[df['title'].str.lower() == potential_title.lower()]
    if matches.empty:
        matches = df[df['title'].str.contains(potential_title, case=False, na=False)]

    if not matches.empty:
        best_match = matches.loc[matches['vote_count'].idxmax()]
        movie_id = int(best_match['id'])
        confirmed_title = best_match['title']
        print(f"  - SUCCESS: Title verified as '{confirmed_title}' (ID: {movie_id}). Routing to TITLE search.")
        return {"request_type": "TITLE", "validated_title_id": movie_id, "extracted_title": confirmed_title}
    else:
        print(f"  - FAILED: Title not found. Falling back to CATEGORY search.")
        return {"request_type": "CATEGORY", "validated_title_id": None}


def search_by_title_node(state: AgentState):
    """Searches for recommendations based on a validated title."""
    print("🎬 Node: search_by_title_node")
    movie_id = state["validated_title_id"]
    
    if not movie_id:
        return {"candidate_list": []}

    try:
        source_movie = df[df['id'] == movie_id].iloc[0]
        source_genres = set(source_movie['genres_list'])
        source_keywords = set(source_movie['keywords_list'])
        
        if not source_genres and not source_keywords:
             return {"candidate_list": []}

        GENRE_WEIGHT = 1.0
        KEYWORD_WEIGHT = 1.5

        def calculate_similarity(row):
            target_genres = set(row['genres_list'])
            target_keywords = set(row['keywords_list'])
            genre_score = len(source_genres.intersection(target_genres))
            keyword_score = len(source_keywords.intersection(target_keywords))
            return (GENRE_WEIGHT * genre_score) + (KEYWORD_WEIGHT * keyword_score)

        df_copy = df.copy()
        df_copy['similarity'] = df_copy.apply(calculate_similarity, axis=1)
        
        recommendations = df_copy[(df_copy['similarity'] > 0) & (df_copy['id'] != movie_id)]
        recommendations = recommendations.sort_values(by=['similarity', 'vote_average'], ascending=False)
        
        candidate_list = recommendations.head(15).to_dict('records')
        print(f"  - Found {len(candidate_list)} recommendations.")
        return {"candidate_list": candidate_list}
    except Exception as e:
        print(f"  - ERROR during DataFrame search: {e}")
        return {"candidate_list": []}


def search_by_category_node(state: AgentState):
    """Searches for recommendations based on genres and keywords."""
    print("🎭 Node: search_by_category_node")
    genres = state.get("extracted_genres", [])
    keywords = state.get("extracted_keywords", [])

    if not genres and not keywords:
        return {"candidate_list": []}
    
    genre_mask = pd.Series(False, index=df.index)
    if genres:
        print(f"  - Searching with Genres: {genres}")
        genres_set = set(g.lower() for g in genres)
        genre_mask = df['genres_list'].apply(lambda lst: not genres_set.isdisjoint(set(g.lower() for g in lst)))
        
    keyword_mask = pd.Series(False, index=df.index)
    if keywords:
        print(f"  - Searching with Keywords: {keywords}")
        keyword_pattern = '|'.join([re.escape(k) for k in keywords])
        keyword_mask = df['keywords'].str.contains(keyword_pattern, case=False, na=False)

    combined_mask = genre_mask | keyword_mask
    candidates_df = df[combined_mask]
    
    candidates_df = candidates_df.sort_values(by=['vote_average', 'popularity'], ascending=False)
    candidate_list = candidates_df.head(20).to_dict('records')
    print(f"  - Found {len(candidate_list)} candidates.")
    return {"candidate_list": candidate_list}


def analyze_candidates_node(state: AgentState):
    """Uses the LLM to analyze candidates and write justifications."""
    print("🧠 Node: analyze_candidates_node")
    question, candidate_list = state["question"], state["candidate_list"]
    
    if not candidate_list:
        error_message = "I couldn't find any relevant movies to analyze."
        if state['request_type'] == 'TITLE':
            error_message = f"I found the movie '{state['extracted_title']}', but I couldn't find any similar movies to recommend."
        return {"analysis_result": json.dumps({"error": error_message})}

    clean_candidate_list = []
    for cand in candidate_list:
        clean_candidate_list.append({
            "title": cand.get("title"),
            "overview": (cand.get('overview', '') or '')[:400] + '...',
            "genres": ', '.join(cand.get('genres_list', [])),
            "keywords": ', '.join(cand.get('keywords_list', [])[:10]),
            "vote_average": cand.get("vote_average")
        })

    prompt_content = f"""You are a charismatic and expert movie recommender. Your goal is to get the user excited about new movies based on their request.

    **User's Original Request:** "{question}"
    **Candidate Movie Data:**
    {json.dumps(clean_candidate_list, indent=2)}
    
    **Instructions:**
    1. Analyze the candidates and select the top 5 best matches.
    2. For "justification", write a short, exciting pitch. Sell the movie!
    3. You MUST respond with ONLY a single, valid JSON object with a key "recommendations".
    4. Each item in the array MUST include "title", "justification", AND "vote_average".

    **EXAMPLE FORMAT:**
    ```json
    {{
      "recommendations": [
        {{ "title": "Example Movie 1", "vote_average": 8.8, "justification": "If you loved the first movie, get ready for a wild ride!" }}
      ]
    }}
    ```
    Your JSON Output:
    """
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    response = text_generation_pipeline(formatted_prompt)[0]['generated_text']
    
    potential_json_str = ""
    markdown_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if markdown_match:
        potential_json_str = markdown_match.group(1).strip()
    else:
        potential_json_str = response.strip()

    try:
        parsed_data = json.loads(potential_json_str)
        if isinstance(parsed_data, list):
            print("  - AI returned a raw array. Wrapping it in the correct object structure.")
            analysis_result = json.dumps({"recommendations": parsed_data})
        elif isinstance(parsed_data, dict) and "recommendations" in parsed_data:
             print("  - Successfully extracted and validated JSON object from LLM.")
             analysis_result = potential_json_str
        else:
            raise ValueError("Parsed JSON is not in the expected format.")
        return {"analysis_result": analysis_result}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  - ERROR: LLM response failed validation. Reason: {e}")
        print(f"  - LLM's raw response snippet: {response[:500]}...")
        return {"analysis_result": json.dumps({"error": "The AI analyst returned a response that was not in a valid or expected format."})}


def format_recommendation_node(state: AgentState):
    """Formats the final output to be user-friendly."""
    print("✍️ Node: format_recommendation_node")
    try:
        data = json.loads(state["analysis_result"])
        
        if "error" in data:
            return {"recommendation": f"I'm sorry, I ran into an issue: {data['error']}"}
            
        final_recs = data.get("recommendations", [])
        if not final_recs:
            return {"recommendation": "After analysis, I couldn't find a strong match for your specific criteria."}
            
        md_string = "Based on your request, here are a few hand-picked recommendations I think you'll love:\n\n"
        for i, rec in enumerate(final_recs, 1):
            title = rec.get('title', 'N/A')
            justification = rec.get('justification', 'No justification provided.')
            score = rec.get('vote_average', 0)
            formatted_score = f"{score:.1f}" if score else "N/A"
            md_string += f"### {i}. {title} (⭐ {formatted_score})\n"
            md_string += f"**Why it's a perfect match:** {justification}\n\n---\n\n"
            
        return {"recommendation": md_string}
    except json.JSONDecodeError:
        return {"recommendation": "I'm sorry, there was an error processing the analysis results."}
