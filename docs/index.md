# Agenty

A Pythonic framework for building AI agents and LLM pipelines, powered by [pydantic-ai](https://github.com/pydantic/pydantic-ai). The framework emphasizes simplicity and maintainability without sacrificing power, making it an ideal choice for both rapid prototyping.

!!! warning
    **Active Development**: Agenty is under active development. Expect frequent breaking changes until we reach a stable release.

Agenty provides a clean, type-safe interface for creating:
- Conversational AI agents with structured inputs and outputs
- LLM pipelines
- Complex agent interactions with minimal boilerplate

## Key Features
- Intuitive Pythonic interfaces that feel natural to use
- Jinja2 templates for prompts and messages for dynamic context
- Automatic conversation history management
- Structured Agent I/O for predictable behavior
- Built on pydantic-ai for type validation

Whether you're building a simple chatbot or a complex multi-agent system, Agenty helps you focus on logic rather than infrastructure.
The framework is currently only officially supported with the OpenAI API (through a proxy such as [LiteLLM](https://docs.litellm.ai/docs/simple_proxy)/[OpenRouter](https://openrouter.ai/docs/quick-start)) although theoretically it supports all the models supported by pydantic-ai.

!!! tip
    Looking for a more mature alternative? Check out [atomic-agents](https://github.com/BrainBlend-AI/atomic-agents), which inspired this project.
