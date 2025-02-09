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

For a complete implementation, see [extract_users.py](https://github.com/jonchun/agenty/blob/main/examples/extract_users.py) in the examples directory.

## Working with Sequences

Extracting a list of data is an extremely common task. You can work with sequences (lists) of any supported type. Here's an example from [news_pipeline.py](https://github.com/jonchun/agenty/blob/main/examples/news_pipeline.py):

```python
from typing import List
from agenty.types import BaseIO
from agenty import Agent

class Article(BaseIO):
    title: str
    content: str
    source: str

class ArticleExtractor(Agent[str, List[Article]]):
    input_schema = str
    output_schema = List[Article]
    system_prompt = "Extract articles from the news feed."

# Usage
news_feed = """
Breaking: Tech giant launches new AI model
In a surprising move, the company revealed their latest...

Market Update: Stocks reach record high
Global markets continued their upward trend as...
"""
articles = await ArticleExtractor().run(news_feed)

for article in articles:
    print(f"Title: {article.title}")
    print(f"Content: {article.content}")
    print()
```

## Complex Data Structures

You can create nested structures for more complex data. Here's an example from [nested_data_structures.py](https://github.com/jonchun/agenty/blob/main/examples/nested_data_structures.py):

```python
from typing import List, Optional
from agenty.types import BaseIO

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
text = """
John Smith, 34, lives at 123 Main St, Boston, USA 02108.
He can be reached at john@email.com or (555) 123-4567.
John enjoys hiking, photography, and cooking.
"""
profile = await ProfileAnalyzer().run(text)

# Access nested data with type safety
print(f"Name: {profile.name}")         # John Smith
print(f"City: {profile.address.city}") # Boston
print(f"Email: {profile.contact.email}") # john@email.com
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

BaseIO objects are Pydantic Models. You can add custom validation to your models:

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
