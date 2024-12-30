from mainframe_orchestra import Task, Agent, Conduct, FileTools, OpenaiModels
import os
import requests

# A receipt processing system that orchestrates agents to:
# 1. Transcribe receipt images
# 2. Save data to CSV files

class ImageTools:
    @classmethod
    def transcribe_image(cls, image_url: str) -> dict:
        """
        Transcribes structured text and data from an image URL.
        Appropriate for images containing structured data such as tables, charts, or other data visualizations.

        Args:
            image_url (str): The URL of the image containing text or structured data.

        Returns:
            dict: The response from OpenAI containing the transcribed structured data.
        """
        try:
            # Prepare the payload for OpenAI API
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe any text and structured data from this image, preserving the formatting and structure."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            # Call OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            # Parse the response
            result = response.json()

            return result

        except Exception as e:
            return {"error": f"Failed to transcribe image: {str(e)}"}


transcription_agent = Agent(
    agent_id="transcription_agent",
    role="Transcription Agent",
    goal="Transcribe the image and return all of the data in the receipt.",
    tools=[ImageTools.transcribe_image],
    llm=OpenaiModels.gpt_4o_mini
)

archivist_agent = Agent(
    agent_id="archivist_agent",
    role="Archivist Agent",
    goal="Archive the transcribed text to a file.",
    attributes="You know to name the file as the store name and date of the receipt if provided.",
    tools=[FileTools.write_csv],
    llm=OpenaiModels.gpt_4o
)

coordinator_agent = Agent(
    agent_id="coordinator_agent",
    role="Coordinator Agent",
    goal="Coordinate the transcription and archiving agents.",
    tools=[Conduct.conduct_tool(transcription_agent, archivist_agent)],
    llm=OpenaiModels.gpt_4o
)

def task(user_input):
    return Task.create(
        agent=coordinator_agent,
        instruction=f"Coordinate your team to assist with the given task: '{user_input}. Make sure they transcribe and save the data to a csv file in entirety."
    )

def main():
    user_input = input("Enter the URL of the receipt image you want to transcribe and save to csv, e.g. image url, google drive image url, etc:\n")
    response = task(user_input)
    print(response)

# example image: https://ocr.space/Content/Images/receipt-ocr-original.webp

if __name__ == "__main__":
    main()