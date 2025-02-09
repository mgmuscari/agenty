# Structured I/O

Agenty provides powerful type-safe input and output handling through structured I/O types. This system enables you to work with complex data structures while maintaining type safety and validation.

## Overview

The structured I/O system is built on two key concepts:

-   `AgentIO`: A union type that supports primitive types and structured objects
-   `BaseIO`: A base class for creating structured data models (built on Pydantic)

Supported types include:

```python
AgentIO = Union[
    bool,          # Boolean values
    int,           # Integer numbers
    float,         # Floating point numbers
    str,           # Text strings
    BaseIO,        # Structured data models
    Sequence[...], # Lists/sequences of any above type
]
```

## Basic Usage

Here's a simple example of using structured I/O with a user information extractor:

```python
from typing import List
from agenty import Agent
from agenty.types import BaseIO

class User(BaseIO):
    name: str
    age: int
    hobbies: List[str]

class UserExtractor(Agent[str, List[User]]):
    input_schema = str
    output_schema = List[User]
    system_prompt = "Extract user information from the text"
```

## Working with Sequences

Extracting a list of data is an extremely common task. You can work with sequences (lists) of any supported type:

```python
from typing import List, Dict
from agenty.types import BaseIO

class NewsArticle(BaseIO):
    title: str
    content: str
    tags: List[str]

class NewsAggregator(Agent[str, List[NewsArticle]]):
    input_schema = str
    output_schema = List[NewsArticle]
    system_prompt = "Extract news articles from the text"

async def main():
    agent = NewsAggregator()
    text = """
    Breaking: New AI Breakthrough
    Scientists announce major progress in machine learning...
    #tech #ai #research

    Weather Update: Storm Warning
    Coastal areas prepare for incoming storm system...
    #weather #safety
    """
    articles = await agent.run(text)

    for article in articles:
        print(f"Title: {article.title}")
        print(f"Tags: {', '.join(article.tags)}")
        print()
```

## Complex Data

You can create nested structures for more complex data:

```python
from typing import List, Optional
from agenty.types import BaseIO
from agenty.agent import Agent

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
    system_prompt = "Extract detailed profile information"

# Example usage:
"""
profile = await ProfileAnalyzer().run(
    '''John Smith, 34, lives at 123 Main St, Boston, USA 02108.
    He can be reached at john@email.com or (555) 123-4567.
    John enjoys hiking, photography, and cooking.'''
)

# Access nested data with type safety
print(f"Name: {profile.name}")         # John Smith
print(f"City: {profile.address.city}") # Boston
print(f"Email: {profile.contact.email}") # john@email.com
"""
```

## Transformers

Transformers are lightweight agents that convert data between different types in a type-safe manner. They are particularly useful in pipelines where you need to transform data before or after processing. Simply override the `transform()` method and handle the transformation between data types.

```python
from typing import Any
from agenty import Transformer, Agent

# Transform any input to integer
class IntTransformer(Transformer[Any, int]):
    async def transform(
        self,
        input_data: Any,
    ) -> int:
        if input_data is None:
            raise ValueError("Input data is required")
        return int(input_data)

# Create a reusable transformer
to_int = IntTransformer()

# Agent that processes numbers
class NumberProcessor(Agent[int, str]):
    input_schema = int
    output_schema = str
    system_prompt = "Convert the number to words"

# Create pipeline and run synchronously
number_processor = to_int | NumberProcessor()
result = number_processor.run_sync("123")
```

!!! note

    The above trivial example is purely for demonstration. Pydantic will actually attempt to automatically convert between compatible types and a `to_int()` transformer without any additional logic is not actually very useful.

## Type Safety and Validation

The structured I/O system provides 2 layers of type safety:

1. **Static Type Checking**: Generics ensure your type checker works as expected:

```python
class DataProcessor(Agent[List[int], float]):
    input_schema = List[int]   # Must match first type parameter
    output_schema = float      # Must match second type parameter
```

2. **Runtime Validation**: Pydantic-based validation ensures data correctness:

```python
class Temperature(BaseIO):
    celsius: float

    @property
    def fahrenheit(self) -> float:
        return (self.celsius * 9/5) + 32

    @property
    def kelvin(self) -> float:
        return self.celsius + 273.15

class WeatherStation(Agent[str, Temperature]):
    input_schema = str
    output_schema = Temperature

    # Invalid data will raise validation errors
    # e.g., if the model returns non-numeric temperature
```

## Best Practices

1. **Type Safety**

    - Always specify input_schema and output_schema
    - Use type hints consistently
    - Let static type checkers help catch errors early
    - Handle validation errors gracefully

2. **Pydantic Model Design**

    - Keep models focused and single-purpose
    - Use descriptive field names
    - Include type hints for all fields
    - Use Optional[] for fields that might be missing
    - Consider adding validation methods or properties

3. **Data Validation**

    - Add field validators when needed
    - Use Pydantic's built-in validation features
    - Consider adding custom validation methods
    - Document any validation requirements

4. **Error Handling**
    - Plan for validation errors
    - Provide meaningful error messages
    - Consider adding custom error types
    - Document expected error cases

## Advanced Features

### Custom Validation

You can add custom validation to your models:

```python
from pydantic import validator
from agenty.types import BaseIO

class User(BaseIO):
    name: str
    age: int
    email: str

    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
```

### Rich Console Output

BaseIO models automatically support rich console output:

```python
from agenty.types import BaseIO
from rich.console import Console
class User(BaseIO):
    name: str
    age: int
    email: str
console = Console()
user = User(name="Alice", age=30, email="alice@example.com")
console.print(user)
# pretty colored output
# {
#   "name": "Alice",
#   "age": 30,
#   "email": "alice@example.com"
# }
```
