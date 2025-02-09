import asyncio
import os
from typing import List, Optional
from agenty.types import BaseIO
from agenty import Agent
from agenty.models import OpenAIModel
from rich.console import Console

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


class Address(BaseIO):
    street: str
    city: str
    country: str
    postal_code: str


class Contact(BaseIO):
    email: str
    phone: Optional[str] = None


class Person(BaseIO):
    name: str
    age: int
    address: Address
    contact: Contact
    interests: List[str]


class ProfileAnalyzer(Agent[str, Person]):
    input_schema = str
    output_schema = Person
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Extract detailed profile information"


# Usage
async def main() -> None:
    console = Console()
    agent = ProfileAnalyzer()
    text = """
    John Smith, 34, lives at 123 Main St, Boston, USA 02108.
    He can be reached at john@email.com or (555) 123-4567.
    John enjoys hiking, photography, and cooking.
    """
    profile = await agent.run(text)

    # Access nested data with type safety
    console.print(f"Name: {profile.name}")
    console.print(f"City: {profile.address.city}")
    console.print(f"Email: {profile.contact.email}")
    console.print(profile)
    # Example Output:
    #
    # Name: John Smith
    # City: Boston
    # Email: john@email.com
    # {
    #   "name": "John Smith",
    #   "age": 34,
    #   "address": {
    #     "street": "123 Main St",
    #     "city": "Boston",
    #     "country": "USA",
    #     "postal_code": "02108"
    #   },
    #   "contact": {
    #     "email": "john@email.com",
    #     "phone": "(555) 123-4567"
    #   },
    #   "interests": [
    #     "hiking",
    #     "photography",
    #     "cooking"
    #   ]
    # }


if __name__ == "__main__":
    asyncio.run(main())
