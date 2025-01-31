from collections.abc import MutableMapping

from pydantic_ai.usage import Usage, UsageLimits


class AgentUsage(MutableMapping[str, Usage]):
    """Tracks usage statistics for multiple models in an agent.

    A dictionary-like container that maps model names to their usage statistics.
    Automatically creates new Usage entries for unknown models and provides
    aggregated statistics across all models.
    """

    def __init__(self):
        """Initialize an empty usage tracker."""
        self._usages: dict[str, Usage] = {}

    def __getitem__(self, key: str) -> Usage:
        """Get usage statistics for a model, creating a new entry if needed.

        Args:
            key: Name of the model

        Returns:
            Usage statistics for the specified model
        """
        try:
            return self._usages[key]
        except KeyError:
            self._usages[key] = Usage()
            return self._usages[key]

    def __setitem__(self, key: str, value: Usage) -> None:
        """Set usage statistics for a model.

        Args:
            key: Name of the model
            value: Usage statistics to set for the model
        """
        self._usages[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove usage statistics for a model.

        Args:
            key: Name of the model to remove
        """
        del self._usages[key]

    def __iter__(self):
        """Iterate over model names."""
        return iter(self._usages)

    def __len__(self) -> int:
        """Get the number of models being tracked.

        Returns:
            Number of models with usage statistics
        """
        return len(self._usages)

    @property
    def requests(self) -> int:
        """Get the total number of requests across all models.

        Returns:
            Total number of API requests made
        """
        return sum(usage.requests for usage in self._usages.values())

    @property
    def request_tokens(self) -> int:
        """Get the total number of request tokens across all models.

        Returns:
            Total number of tokens used in requests
        """
        return sum(usage.request_tokens or 0 for usage in self._usages.values())

    @property
    def response_tokens(self) -> int:
        """Get the total number of response tokens across all models.

        Returns:
            Total number of tokens in model responses
        """
        return sum(usage.response_tokens or 0 for usage in self._usages.values())

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens across all models.

        Returns:
            Total number of tokens used (requests + responses)
        """
        return sum(usage.total_tokens or 0 for usage in self._usages.values())


from pydantic_ai.usage import UsageLimits


class AgentUsageLimits(MutableMapping[str, UsageLimits]):
    """Tracks usage limits for multiple models in an agent.

    A dictionary-like container that maps model names to their UsageLimits.
    Automatically creates new UsageLimits entries for unknown models.
    """

    def __init__(self):
        """Initialize an empty UsageLimits tracker."""
        self._usage_limits: dict[str, UsageLimits] = {}

    def __getitem__(self, key: str) -> UsageLimits:
        """Get usage limits for a model, creating a new entry if needed.

        Args:
            key: Name of the model

        Returns:
            UsageLimits for the specified model
        """
        try:
            return self._usage_limits[key]
        except KeyError:
            self._usage_limits[key] = UsageLimits()
            return self._usage_limits[key]

    def __setitem__(self, key: str, value: UsageLimits) -> None:
        """Set UsageLimits for a model.

        Args:
            key: Name of the model
            value: UsageLimits to set for the model
        """
        self._usage_limits[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove UsageLimits for a model.

        Args:
            key: Name of the model to remove
        """
        del self._usage_limits[key]

    def __iter__(self):
        """Iterate over model names."""
        return iter(self._usage_limits)

    def __len__(self) -> int:
        """Get the number of models being tracked.

        Returns:
            Number of models with UsageLimits
        """
        return len(self._usage_limits)
