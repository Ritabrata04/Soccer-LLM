import json
import datetime
from collections import defaultdict
from tqdm import tqdm
import ollama  # 🔁 NEW: Use Ollama

# --- Load only captions ---
with open("Labels-caption-2.json") as f:
    captions_data = json.load(f)

captions = captions_data["annotations"]  # use only the captions

# --- Convert game time (e.g., '2 - 12:20') to seconds
def game_time_to_seconds(gt):
    half, mmss = gt.split(" - ")
    m, s = map(int, mmss.split(":"))
    return (int(half) - 1) * 45 * 60 + m * 60 + s

# --- Build timeline only from captions
timeline = defaultdict(list)

for caption in captions:
    if isinstance(caption, dict) and "gameTime" in caption and "description" in caption:
        sec = game_time_to_seconds(caption["gameTime"])
        timeline[sec].append(f"[commentary] {caption['description']}")

# --- Create sliding windows
WINDOW_SIZE = 60
STEP_SIZE = 20
events_sorted = sorted(timeline.keys())
start_times = list(range(min(events_sorted), max(events_sorted), STEP_SIZE))

windows = []
for start in start_times:
    end = start + WINDOW_SIZE
    content = []
    for t in range(start, end + 1):
        content.extend(timeline.get(t, []))
    if content:
        windows.append({
            "start_time": str(datetime.timedelta(seconds=start)),
            "end_time": str(datetime.timedelta(seconds=end)),
            "events": content
        })

# --- Judge prompts (longer system-user role type prompts)
def prompt_judge_1(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 1, an expert in identifying crucial and game-changing moments in football matches. "
            "Your task is to evaluate whether a time window contains important events like goals, penalties, red cards, or other key actions that directly impact the outcome of the match. "
            "These are events that change the course of the game, often leading to a score, or major turning points that influence the result of the game. "
            "Please evaluate whether the window contains any such events, and provide a YES or NO response. "
            "If you decide it is a key moment, briefly explain why this moment is significant to the game."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Does this time window contain an important event that could change the game outcome, such as a goal, red card, or penalty? "
            "Respond with YES or NO and provide a concise justification for your decision."
        )
    }]

def prompt_judge_2(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 2, a specialist in identifying emotionally significant moments in a football match. "
            "This includes high-drama moments such as spectacular saves, near-misses, controversial moments (such as VAR decisions), or intense crowd reactions. "
            "Your role is to detect emotionally charged moments that might excite fans, players, and commentators. "
            "These moments may not necessarily change the outcome of the game, but they are moments that stand out due to their emotional or dramatic nature. "
            "Please evaluate the time window and decide whether it contains such an emotionally significant event. "
            "Provide a YES or NO answer, followed by a brief explanation of why this moment is emotionally significant."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Is this an emotionally significant highlight that would excite fans or viewers, such as a dramatic save, near-miss, or controversial moment? "
            "Respond with YES or NO, and provide a brief explanation for your answer."
        )
    }]

def prompt_judge_3(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 3, a tactical analyst specializing in evaluating key strategic moments in a football match. "
            "This includes sequences that show a change in the momentum of the game, impactful substitutions, tactical shifts like formation changes, or significant build-up play leading to a scoring opportunity. "
            "Your task is to assess whether a given time window contains tactical moments that could influence the overall flow and outcome of the match, even if they don't result in a direct score. "
            "Please decide whether this time window contains such tactical moments and provide a YES or NO answer, with a brief explanation of the tactical significance."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Does this window contain significant tactical moments, such as a momentum shift, impactful substitution, or build-up play? "
            "Respond with YES or NO, and explain why this moment is tactically important."
        )
    }]

# --- Main loop
highlight_segments = []
highlight_segments_with_event = []
judges = [prompt_judge_1, prompt_judge_2, prompt_judge_3]

for win in tqdm(windows, desc="Evaluating highlights", ncols=100):
    votes = 0
    for judge_fn in judges:
        prompt = judge_fn(win)
        try:
            response = ollama.chat(model="llama3", messages=prompt)  # Use Llama3 model for evaluation
            answer = response["message"]["content"].strip().lower()
            if "yes" in answer:
                votes += 1
        except Exception as e:
            print(f"Judge error: {e}")
    if votes >= 2:
        highlight_segments.append({
            "start_time": win["start_time"],
            "end_time": win["end_time"],
            "description": f"Selected by {votes}/3 Llama3 judges"
        })

        # Summarize event
        try:
            event_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional football commentator. Your job is to briefly summarize the most significant moment in a match window "
                        "based on the commentary lines provided. Be concise (1 sentence max), and use real-world football language (e.g., 'Goal by X', 'Save by Y')."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Match commentary from {win['start_time']} to {win['end_time']}:\n"
                        f"{win['events']}\n\n"
                        "Summarize the key event in this window."
                    )
                }
            ]
            event_response = ollama.chat(model="llama3", messages=event_prompt)  # Use Llama3 model for summarization
            event_description = event_response["message"]["content"].strip()
        except Exception as e:
            print(f"Event summarization failed: {e}")
            event_description = "Unknown event (error)"

        highlight_segments_with_event.append({
            "start_time": win["start_time"],
            "end_time": win["end_time"],
            "event": event_description
        })

# --- Save both outputs
with open("highlight_segments_llama3.json", "w") as f:
    json.dump(highlight_segments, f, indent=2)

with open("highlight_events_llama3.json", "w") as f:
    json.dump(highlight_segments_with_event, f, indent=2)

print("✅ Done! Saved:")
print(" - highlight_segments_llama3.json")
print(" - highlight_events_llama3.json")
