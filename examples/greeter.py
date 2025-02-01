import asyncio
import os

from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


class Greeter(Agent):
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = """You are a greeter. You speak in a {{TONE}} tone. Your response length should be {{RESPONSE_LENGTH}}."""
    TONE: str = "friendly"
    RESPONSE_LENGTH: str = "medium"


async def main():
    agent = Greeter()
    print("# Friendly and medium")
    response = await agent.run("Hello, please greet me!")
    print(response)
    response = await agent.run("Tell me about Los Angeles.")
    print(response)
    print("")
    print("# Angry and short")
    agent.TONE = "angry"
    agent.RESPONSE_LENGTH = "very short"
    response = await agent.run("Tell me more more about Los Angeles.")
    print(response)
    # Sample Output:
    # # Friendly and medium
    # Hello there! 👋 It's wonderful to meet you! How's your day going so far? 😊
    # Los Angeles, often known as the City of Angels, is a vibrant and diverse city located in Southern California. It's renowned for its thriving entertainment industry, being home to Hollywood, where movie and television magic happens. L.A. is famous for its beautiful beaches, such as Santa Monica and Venice Beach, offering stunning sunset views and vibrant boardwalks.

    # Beyond the glitz and glamour, Los Angeles is a cultural melting pot with a rich tapestry of neighborhoods, each offering its unique flavor—from the historic streets of Olvera Street to the trendy shops of Silver Lake. The city also boasts a wide array of museums, art galleries, and theaters, making it a cultural hotspot.

    # Nature enthusiasts will enjoy the scenic hikes in the nearby Santa Monica Mountains or Griffith Park, which also hosts the iconic Griffith Observatory. And let's not forget the diverse culinary scene, featuring everything from gourmet restaurants to delicious food trucks offering cuisines from around the world.

    # Whether you're exploring the Walk of Fame, catching a concert at the Hollywood Bowl, or simply enjoying the sunny weather, Los Angeles is a place full of exciting opportunities and endless things to discover! 🌟🏖️🌆

    # # Angry and short
    # It's big. It's busy. It's full of traffic! That's L.A.


asyncio.run(main())
