# Intelligence Layer Prompts

Prompt templates are stored as Jinja2 `.j2` files and versioned by operator.

Layout:

```
src/intelligence_layer_prompts/{operator_name}/{version}/*.j2
```

Example:

```
src/intelligence_layer_prompts/Email.ReviewDraft/1.0.0/review_draft.j2
```

Use `PromptTemplateLoader` to render templates with strict undefined variables.
