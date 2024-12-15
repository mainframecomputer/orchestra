# Audio Tools

The Audio Tools module provides two main classes: `TextToSpeechTools` and `WhisperTools`. These classes offer functionality for text-to-speech conversion and audio transcription/translation. They are designed to simplify the process of working with audio in your Orchestra applications, handling API interactions and audio processing internally.

## TextToSpeechTools

The `TextToSpeechTools` class provides methods for converting text to speech using either the ElevenLabs API or the OpenAI API.

### Class Methods

#### elevenlabs_text_to_speech()

Converts text to speech using the ElevenLabs API and either plays the generated audio or saves it to a file.

```python
TextToSpeechTools.elevenlabs_text_to_speech(
    text="Hello, world!",
    voice="Giovanni",
    output_file="output.mp3"
)
```

#### openai_text_to_speech()

Generates speech from text using the OpenAI API and either saves it to a file or plays it aloud.

```python
TextToSpeechTools.openai_text_to_speech(
    text="Hello, world!",
    voice="onyx",
    output_file="output.mp3"
)
```

## WhisperTools

The `WhisperTools` class provides methods for transcribing and translating audio using the OpenAI Whisper API.

### Class Methods

#### whisper_transcribe_audio()

Transcribes audio using the OpenAI Whisper API.

```python
WhisperTools.whisper_transcribe_audio(
    audio_input="audio.mp3",
    model="whisper-1",
    language="en",
    response_format="json",
    temperature=0
)
```

#### whisper_translate_audio()

Translates audio using the OpenAI Whisper API.

```python
WhisperTools.whisper_translate_audio(
    audio_input="audio.mp3",
    model="whisper-1",
    response_format="json",
)
```

## Usage in Orchestra

These audio tools can be integrated into your Orchestra agents to enable them to self determine when to speak:

```python
voice_agent = Agent(
    role="Voice Assistant",
    goal="Assist users through voice interaction",
    attributes="Clear speaking voice, good listener, multilingual",
    tools={
        TextToSpeechTools.openai_text_to_speech
    },
    llm=OpenrouterModels.haiku
)
```

This voice agent can use the audio tools to respond with generated speech. The TextToSpeechTools can be used to give agents the ability to initiate speech output at their discretion.

## Direct Usage

You can also use these tools in your main flow to handle audio inputs and outputs from agents and tasks:

```python
def main():
    conversation_history = []
    temp_dir = tempfile.mkdtemp()
    audio_file = os.path.join(temp_dir, "recorded_audio.wav")

    print("Enter 'q' to quit.")

    while True:
        user_input = input("Press Enter to start recording (or 'q' to quit): ").lower()
        if user_input == 'q':
            print("Exiting...")
            break

        # Record user input via microphone upon 'enter'
        RecordingTools.record_audio(audio_file)
        
        # Transcribe the audio
        transcription = WhisperTools.whisper_transcribe_audio(audio_file)

        # Collect the text from the transcription
        user_message = transcription['text'] if isinstance(transcription, dict) else transcription

        # Add the user's message to the conversation history
        conversation_history.append(f"User: {user_message}")

        # Agent acts and responds to the user
        response = respond_task(user_message, conversation_history)
        conversation_history.append(f"Assistant: {response}")
        print("Assistant's response:")
        print(response)

        # Read the agent response out loud
        TextToSpeechTools.elevenlabs_text_to_speech(text=response)

    # Clean up the temporary directory
    os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
```

## Usage Notes

- To use the `TextToSpeechTools` and `WhisperTools` classes, you must set the appropriate API keys in your environment variables (ELEVENLABS_API_KEY and OPENAI_API_KEY).
- The `elevenlabs_text_to_speech` method requires the `elevenlabs` library to be installed. If not present, it will raise an ImportError with instructions to install.
- The `openai_text_to_speech` method uses `pygame` for audio playback. Ensure it's installed in your environment.
- All methods in these classes handle API authentication internally, abstracting away the complexity of token management.
- Error handling is built into these methods, with specific exceptions raised for common issues like missing API keys or module import errors.

By incorporating these audio tools into your Orchestra agents, you can get started with creating sophisticated voice-enabled applications and multi-modal AI assistants capable of processing and generating both text and speech.