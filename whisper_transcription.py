import whisper
import moviepy.editor as mp
import json

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from the given video file and saves it to a specified audio file path.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_path):
    """
    Transcribes the given audio file using Whisper model and returns the segmented transcription.
    """
    model = whisper.load_model("base")  # You can choose a different model like "small", "medium", "large"
    result = model.transcribe(audio_path)
    return result['segments']

# Function to chunk the transcribed sentences and save them as JSON
def save_transcription_as_json(transcription, output_path):
    """
    Saves the transcription of the audio as a JSON file with chunks of sentences.
    """
    chunked_sentences = []
    for segment in transcription:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        chunked_sentences.append({
            "start_time": start_time,
            "end_time": end_time,
            "text": text
        })
    
    # Save the result as a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(chunked_sentences, json_file, indent=4)

# Main function to process video and save transcription
def process_video(video_path, audio_path, json_output_path):
    """
    Extracts audio from the video, transcribes it, and saves the transcription in JSON format.
    """
    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    
    # Transcribe the audio
    transcription = transcribe_audio(audio_path)
    
    # Save the transcription as a JSON file
    save_transcription_as_json(transcription, json_output_path)
    print(f"Transcription saved to {json_output_path}")

# Example usage
if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"  # Replace with your video file path
    audio_path = "extracted_audio.wav"     # Temporary path for the extracted audio
    json_output_path = "transcription.json"  # Path to save the transcription JSON

    process_video(video_path, audio_path, json_output_path)
