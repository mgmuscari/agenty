# Tools

Tools enable agents to interact with their environment by providing access to external functions and capabilities. They are implemented using a simple decorator pattern that automatically registers methods as available tools for your agent.

## Basic Usage

Here's a simple example of implementing tools in an agent:

```python
from agenty import Agent, tool
from agenty.models import OpenAIModel

class WeatherAgent(Agent):
    model = OpenAIModel("gpt-4o", api_key="your-api-key")
    system_prompt = "You help users check the weather and get clothing recommendations."

    def __init__(self, location: str, **kwargs):
        super().__init__(**kwargs)
        self.location = location
        self.temperature = 72  # Simulated temperature

    @tool
    def get_temperature(self) -> float:
        """Get the current temperature for the configured location."""
        return self.temperature

    @tool
    def get_location(self) -> str:
        """Get the currently configured location."""
        return self.location
```

## Type Safety

Tools support type hints which help ensure correct usage and enable better IDE integration.

```python
from typing import List, Dict
from agenty.types import BaseIO

class InventoryItem(BaseIO):
    name: str
    quantity: int

class InventoryAgent(Agent):
    @tool
    def add_item(self, name: str, quantity: int) -> bool:
        """
        Add an item to inventory.

        Args:
            name: The name of the item
            quantity: How many to add
        """
        # Implementation here
        return True

    @tool
    def get_items(self) -> List[InventoryItem]:
        """Get all items in inventory."""
        return [
            InventoryItem(name="apple", quantity=5),
            InventoryItem(name="orange", quantity=3)
        ]
```

## Advanced Usage

Tools can be used for more complex operations and can work with structured data. Here's an example demonstrating async tools and optional parameters:

```python
import asyncio
from datetime import datetime
from typing import Optional, List, Dict
from agenty.types import BaseIO

class Appointment(BaseIO):
    date: datetime
    description: str
    duration_minutes: int
    attendees: Optional[List[str]] = None

class SchedulerAgent(Agent):
    def __init__(self):
        super().__init__()
        self.appointments: Dict[datetime, Appointment] = {}

    @tool
    async def schedule_appointment(
        self,
        date: datetime,
        description: str,
        duration_minutes: int = 30,
        attendees: Optional[List[str]] = None
    ) -> bool:
        """
        Schedule a new appointment.

        Args:
            date: When the appointment should occur
            description: What the appointment is for
            duration_minutes: How long the appointment will last
            attendees: Optional list of attendee email addresses
        """
        if date in self.appointments:
            return False

        self.appointments[date] = Appointment(
            date=date,
            description=description,
            duration_minutes=duration_minutes,
            attendees=attendees or []
        )
        return True

    @tool
    def list_appointments(self, date: Optional[datetime] = None) -> List[Appointment]:
        """
        List appointments, optionally filtered by date.

        Args:
            date: Optional date to filter appointments
        """
        if date:
            return [
                appointment
                for dt, appointment in self.appointments.items()
                if dt.date() == date.date()
            ]
        return list(self.appointments.values())
```

## Example: Roulette

```python
import random
from typing import Tuple, Any
from agenty import Agent, tool
from agenty.types import BaseIO
from agenty.models import OpenAIModel

class PlayerInfo(BaseIO):
    name: str
    balance: float

class RouletteAgent(Agent):
    input_schema = str
    output_schema = str
    model = OpenAIModel("gpt-4o", api_key="your-api-key")
    system_prompt = (
        "You're running a roulette game. Help users place bets and "
        "determine if they've won based on the wheel spin. Use the "
        "player's name in responses to make it more personal."
    )

    def __init__(
        self,
        player_name: str,
        starting_balance: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.player_name = player_name
        self.balance = starting_balance
        self.current_bet = 0.0
        self.bet_number = None

    @tool
    def get_player_info(self) -> PlayerInfo:
        """Get the current player's information including name and balance."""
        return PlayerInfo(name=self.player_name, balance=self.balance)

    @tool
    def place_bet(self, amount: float, number: int) -> bool:
        """
        Place a bet on a specific number.

        Args:
            amount: How much money to bet
            number: Which number to bet on (0-36)
        """
        if amount > self.balance or not 0 <= number <= 36:
            return False
        self.current_bet = amount
        self.bet_number = number
        self.balance -= amount
        return True

    @tool
    def spin_wheel(self) -> Tuple[int, bool]:
        """
        Spin the wheel and return (winning number, whether player won).

        Returns:
            A tuple of (winning number, whether player won)
        """
        winning_number = random.randint(0, 36)
        won = False

        if self.current_bet > 0 and self.bet_number == winning_number:
            self.balance += self.current_bet * 35
            won = True
        self.current_bet = 0
        self.bet_number = None

        return winning_number, won
```

For a complete runnable implementation with a rich console interface and proper configuration, see the [roulette agent example](https://github.com/jonchun/agenty/blob/main/examples/roulette_agent.py) in the repository.

## Best Practices

1. **Clear Documentation**

    - Write detailed docstrings for each tool
    - Include parameter descriptions and return value information
    - Document any side effects or important behaviors

2. **Type Safety**

    - Use type hints for all parameters and return values
    - Leverage `agenty.types.BaseIO` for structured data
    - Validate input parameters when necessary

3. **Error Handling**

    - Return clear success/failure indicators
    - Handle edge cases gracefully
    - Provide meaningful error messages or status codes

4. **State Management**

    - Keep track of state changes in instance variables
    - Consider thread safety for shared resources
    - Reset state appropriately between operations

5. **Tool Design**
    - Keep tools focused and single-purpose
    - Use descriptive names that indicate the tool's function
    - Consider breaking complex operations into multiple tools
    - Make tools reusable across different agent implementations

You can read more about the underlying implementation in the [pydantic-ai function tools documentation](https://ai.pydantic.dev/tools/).
