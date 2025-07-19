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
3. By default, Braintrust is automatically enabled when the API key is present

To explicitly enable Braintrust integration:
```
export BRAINTRUST_ORCHESTRA_ENABLED=true
```

To disable Braintrust integration inside Orchestra:
```
export BRAINTRUST_ORCHESTRA_ENABLED=false
```

You can also use `0` or `no` as values to disable it, or `1` or `yes` to enable it. Braintrust integration is automatically enabled when:
1. The `braintrust` package is installed
2. The `BRAINTRUST_API_KEY` environment variable is set with a valid API key
3. The `BRAINTRUST_ORCHESTRA_ENABLED` environment variable is either:
   - Set to enable it (`true`, `1`, or `yes`)
   - Not set at all (defaults to enabled when API key exists)

When enabled, Braintrust will trace all tool calls and LLM requests made through Orchestra's unified LiteLLM interface. This includes requests to all supported providers such as OpenAI, OpenRouter, Groq, Together AI, Deepseek, Anthropic, Google, and others, providing detailed logs and analytics in your Braintrust dashboard.

### LiteLLM Integration Benefits for Observability

Orchestra v1.0.0's unified LiteLLM architecture provides enhanced observability benefits:

- **Consistent Logging**: All LLM requests are logged in a standardized format regardless of provider
- **Unified Metrics**: Performance metrics are comparable across different providers
- **Simplified Monitoring**: Single integration point for observability tools
- **Enhanced Error Tracking**: Standardized error handling and reporting across all providers
