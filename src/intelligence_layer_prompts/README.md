# Intelligence Layer Prompts

Prompt templates are stored as Jinja2 `.j2` files and versioned by operator.

Layout:

```
src/intelligence_layer_prompts/*.j2
```

Example:

```
src/intelligence_layer_prompts/email_review_draft.v1.j2
```

Use `PromptTemplateLoader` to render templates with strict undefined variables.
