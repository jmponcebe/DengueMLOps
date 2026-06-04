# Development Approach — Reference

## Philosophy

Start with exploratory notebooks for rapid prototyping, then incrementally transfer logic to structured production code.

## Workflow

1. **Notebooks** (`notebooks/`): rapid prototyping, EDA, experimentation. Keep concise with practical code and minimal explanations.
2. **Production code** (`src/`): structured modules organized by functionality (data, features, models, monitoring). Each module is independently testable.
3. **Documentation** (`docs/`): comprehensive explanations, methodology, and theory. Separate from code.
4. **TFM thesis** (`docs/memoria/`): final academic content. CIDaeN UCLM template.

## Separation Principle

- Code for doing
- Docs for understanding
- Memoria for academic submission

## Code Style Target

Code should appear written by an MLOps engineer with 3-5 years experience:
- Pragmatic solutions over theoretical elegance
- No over-engineering
- Natural imperfections (small inconsistencies that reflect human development)
- Brief, useful comments (focus on "why" not "what")
