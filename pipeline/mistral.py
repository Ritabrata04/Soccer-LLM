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

# --- Judge prompts (same logic, focused on commentary)
def prompt_judge_1(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 1, an expert in identifying game-changing moments in football matches. "
            "Evaluate whether this time window contains important events like goals, red cards, or penalties. "
            "Return YES or NO, followed by a short reason."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Should this window be highlighted? Respond with YES or NO and a brief justification."
        )
    }]

def prompt_judge_2(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 2, a specialist in emotionally exciting moments like big saves, near misses, or VAR drama. "
            "Return YES or NO and explain briefly."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Is this an exciting highlight? Respond with YES or NO and a brief reason."
        )
    }]

def prompt_judge_3(window):
    return [{
        "role": "system",
        "content": (
            "You are LLM Judge 3, a tactical analyst. Highlight sequences that show momentum shifts, impactful substitutions, or coordinated build-up. "
            "Respond with YES or NO and a brief explanation."
        )
    }, {
        "role": "user",
        "content": (
            f"Time Window: {window['start_time']} to {window['end_time']}\n"
            f"Commentary:\n{window['events']}\n\n"
            "Should this window be included based on tactical significance? Respond with YES or NO and explain briefly."
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
            response = ollama.chat(model="mistral", messages=prompt)  # Switch to Mistral model
            answer = response["message"]["content"].strip().lower()
            if "yes" in answer:
                votes += 1
        except Exception as e:
            print(f"Judge error: {e}")
    if votes >= 2:
        highlight_segments.append({
            "start_time": win["start_time"],
            "end_time": win["end_time"],
            "description": f"Selected by {votes}/3 Mistral judges"
        })

        # Summarize event
        try:
            event_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional football commentator. Your job is to briefly summarize the most significant moment in a match window "
                        "based on the commentary lines provided. Be concise (1 sentence max), use real-world football language (e.g. 'Goal by X', 'Save by Y')."
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
            event_response = ollama.chat(model="mistral", messages=event_prompt)  # Switch to Mistral model
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
with open("highlight_segments_mistral.json", "w") as f:
    json.dump(highlight_segments, f, indent=2)

with open("highlight_events_mistral.json", "w") as f:
    json.dump(highlight_segments_with_event, f, indent=2)

print("✅ Done! Saved:")
print(" - highlight_segments_mistral.json")
print(" - highlight_events_mistral.json")
