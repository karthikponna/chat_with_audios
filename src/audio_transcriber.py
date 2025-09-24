import assemblyai as aai
from typing import List, Dict
import os


class Transcribe:
    """
    Class to transcribe audio files with speaker labeling using AssemblyAI.
    """
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """
        Transcribe an audio file and return speaker-labeled transcripts.
        """
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2  
        )
        transcript = self.transcriber.transcribe(audio_path, config=config)
        speaker_transcripts = []
        for utterance in transcript.utterances:
            speaker_transcripts.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": utterance.text
            })
        return speaker_transcripts