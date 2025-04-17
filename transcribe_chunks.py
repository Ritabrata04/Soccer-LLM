import whisper
import moviepy.editor as mp
import json
import os

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from the given video file and saves it to a specified audio file path.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

# Function to transcribe audio to text using Whisper
def transcribe_audio_in_chunks(audio_path, window_size=8):
    """
    Transcribes the given audio file in static chunks (e.g., 8 seconds).
    """
    model = whisper.load_model("base")  # You can choose a different model like "small", "medium", "large"
    
    # Load the audio file
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # Get the audio duration
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    
    # Process audio in fixed windows of `window_size` seconds
    segments = []
    for start_time in range(0, int(duration), window_size):
        # Define the end time for the chunk (but not exceeding the total duration)
        end_time = min(start_time + window_size, duration)
        
        # Extract the segment of the audio
        segment = audio[int(start_time * whisper.audio.SAMPLE_RATE):int(end_time * whisper.audio.SAMPLE_RATE)]
        
        # Transcribe the segment using Whisper
        result = model.transcribe(segment)
        
        # Append the result as a segment with start and end time
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": result['text']
        })
    
    return segments

# Function to save the transcription as JSON
def save_transcription_as_json(transcription, output_path):
    """
    Saves the transcription of the audio as a JSON file with chunks of sentences.
    """
    with open(output_path, 'w') as json_file:
        json.dump(transcription, json_file, indent=4)

# Main function to process video and save transcription
def process_video(video_path, audio_path, json_output_path, window_size=8):
    """
    Extracts audio from the video, transcribes it in static windows, and saves the transcription in JSON format.
    """
    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    
    # Transcribe the audio in fixed windows
    transcription = transcribe_audio_in_chunks(audio_path, window_size)
    
    # Save the transcription as a JSON file
    save_transcription_as_json(transcription, json_output_path)
    print(f"Transcription saved to {json_output_path}")

# Example usage
if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"  # Replace with your video file path
    audio_path = "extracted_audio.wav"     # Temporary path for the extracted audio
    json_output_path = "transcription.json"  # Path to save the transcription JSON

    process_video(video_path, audio_path, json_output_path)
