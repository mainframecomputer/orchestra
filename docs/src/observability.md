# Observability

### Braintrust Integration

##### How do I enable or disable Braintrust integration?

Orchestra has built-in support for [Braintrust](https://www.braintrust.dev/), which provides observability and evaluation for LLM applications.

To enable Braintrust integration:
1. Install the `braintrust` package: `pip install braintrust`
2. Set your Braintrust API key as an environment variable:
   ```
   export BRAINTRUST_API_KEY=your_api_key
   ```
3. Ensure the `BRAINTRUST_ORCHESTRA_DISABLED` environment variable is not set, or is set to False

To disable Braintrust integration inside Orchestra:
```
export BRAINTRUST_ORCHESTRA_DISABLED=true
```

You can also use `1` or `yes` as values to disable it. Braintrust integration is automatically enabled when:
1. The `braintrust` package is installed
2. The `BRAINTRUST_API_KEY` environment variable is set with a valid API key
3. The `BRAINTRUST_ORCHESTRA_DISABLED` environment variable is not set to disable it

When enabled, Braintrust will trace all tool calls and requests to OpenAI, OpenRouter, Groq, Together AI, and Deepseek, providing detailed logs and analytics in your Braintrust dashboard.