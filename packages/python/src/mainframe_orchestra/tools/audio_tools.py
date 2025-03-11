# Copyright 2024 Mainframe-Orchestra Contributors. Licensed under Apache License 2.0.

import os
from typing import List, Dict, Optional, Literal, Union
from dotenv import load_dotenv
import requests
from ..utils.braintrust_utils import traced
from openai import OpenAI
import time
import io

load_dotenv()

def check_elevenlabs():
    try:
        from elevenlabs import play
        from elevenlabs.client import ElevenLabs
        return play, ElevenLabs
    except ImportError:
        raise ImportError("elevenlabs is required for text_to_speech. Install with `pip install elevenlabs`")

def check_pygame():
    try:
        import os
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        import pygame
        return pygame
    except ImportError:
        raise ImportError("pygame is required for audio playback. Install with `pip install pygame`")

class TextToSpeechTools:
    @traced(type="tool")
    @staticmethod
    def elevenlabs_text_to_speech(text: str, voice: str = "Giovanni", output_file: str = None):
        """
        Convert text to speech using the ElevenLabs API and either play the generated audio or save it to a file.

        Args:
            text (str): The text to convert to speech.
            voice (str, optional): The name of the voice to use. Defaults to "Giovanni".
            output_file (str, optional): The name of the file to save the generated audio. If None, the audio will be played aloud.

        Returns:
            None
        """
        try:
            from elevenlabs import play
            from elevenlabs.client import ElevenLabs
        except ModuleNotFoundError as e:
            raise ImportError(f"{e.name} is required for text_to_speech. Install with `pip install {e.name}`")

        api_key = os.getenv('ELEVENLABS_API_KEY')

        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

        client = ElevenLabs(api_key=api_key)

        audio = client.generate(
            text=text,
            voice=voice,
            model="eleven_multilingual_v2"
        )

        if output_file:
            with open(output_file, "wb") as file:
                file.write(audio)
        else:
            play(audio)
        return audio

    @traced(type="tool")
    @staticmethod
    def openai_text_to_speech(text: str, voice: str = "onyx", output_file: str = None):
        """
        Generate speech from text using the OpenAI API and either save it to a file or play it aloud.

        Args:
            text (str): The text to convert to speech.
            voice (str, optional): The name of the voice to use. Defaults to "onyx".
            output_file (str, optional): The name of the file to save the generated audio. If None, the audio will be played aloud.

        Returns:
            None
        """
        try:
            import os
            os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
            import pygame
        except ModuleNotFoundError:
            raise ImportError("pygame is required for audio playback in the openai_text_to_speech tool. Install with `pip install pygame`")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.0
        )

        if output_file:
            # Save the audio file using the recommended streaming method
            with open(output_file, "wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        else:
            time.sleep(0.7)
            # Play the audio directly using pygame
            pygame.mixer.init()
            audio_data = b''.join(chunk for chunk in response.iter_bytes())
            audio_file = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_file)

            # Add a small delay before playing
            time.sleep(1)

            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()

class WhisperTools:
    @traced(type="tool")
    @staticmethod
    def whisper_transcribe_audio(
        audio_input: Union[str, List[str]],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[Literal["segment", "word"]]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Transcribe audio using the OpenAI Whisper API.

        Args:
            audio_input (Union[str, List[str]]): Path to audio file(s) or list of paths.
            model (str): The model to use for transcription. Default is "whisper-1".
            language (Optional[str]): The language of the input audio. If None, Whisper will auto-detect.
            prompt (Optional[str]): An optional text to guide the model's style or continue a previous audio segment.
            response_format (str): The format of the transcript output. Default is "json".
            temperature (float): The sampling temperature, between 0 and 1. Default is 0.
            timestamp_granularities (Optional[List[str]]): List of timestamp granularities to include.

        Returns:
            Union[Dict, List[Dict]]: Transcription result(s) in the specified format.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env file")

        url = 'https://api.openai.com/v1/audio/transcriptions'
        headers = {'Authorization': f'Bearer {api_key}'}

        def process_single_file(file_path):
            with open(file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': model,
                    'response_format': response_format,
                    'temperature': temperature,
                }
                if language:
                    data['language'] = language
                if prompt:
                    data['prompt'] = prompt
                if timestamp_granularities:
                    data['timestamp_granularities'] = timestamp_granularities

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()

                if response_format == 'json' or response_format == 'verbose_json':
                    return response.json()
                else:
                    return response.text

        if isinstance(audio_input, str):
            return process_single_file(audio_input)
        elif isinstance(audio_input, list) and all(isinstance(file, str) for file in audio_input):
            return [process_single_file(file) for file in audio_input if os.path.isfile(file)]
        else:
            raise ValueError('Invalid input type. Expected string or list of strings.')

    @traced(type="tool")
    @staticmethod
    def whisper_translate_audio(
        audio_input: Union[str, List[str]],
        model: str = "whisper-1",
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[Literal["segment", "word"]]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Translate audio to English using the OpenAI Whisper API.

        Args:
            audio_input (Union[str, List[str]]): Path to audio file(s) or list of paths.
            model (str): The model to use for translation. Default is "whisper-1".
            prompt (Optional[str]): An optional text to guide the model's style or continue a previous audio segment.
            response_format (str): The format of the transcript output. Default is "json".
            temperature (float): The sampling temperature, between 0 and 1. Default is 0.
            timestamp_granularities (Optional[List[str]]): List of timestamp granularities to include.

        Returns:
            Union[Dict, List[Dict]]: Translation result(s) in the specified format.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env file")

        url = 'https://api.openai.com/v1/audio/translations'
        headers = {'Authorization': f'Bearer {api_key}'}

        def process_single_file(file_path):
            with open(file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': model,
                    'response_format': response_format,
                    'temperature': temperature,
                }
                if prompt:
                    data['prompt'] = prompt
                if timestamp_granularities:
                    data['timestamp_granularities'] = timestamp_granularities

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()

                if response_format == 'json' or response_format == 'verbose_json':
                    return response.json()
                else:
                    return response.text

        if isinstance(audio_input, str):
            return process_single_file(audio_input)
        elif isinstance(audio_input, list):
            return [process_single_file(file) for file in audio_input if os.path.isfile(file)]
        else:
            raise ValueError('Invalid input type. Expected string or list of strings.')
