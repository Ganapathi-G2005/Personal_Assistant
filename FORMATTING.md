# Sidekick Formatting Guide

This project follows a **human-curated formatting style** inspired by the Derma project. Tooling helps with consistency, but readability and practical structure stay the priority.

## 1) Tooling Philosophy

- Use formatter output as a baseline, not the only source of truth.
- Use ESLint primarily for **correctness** (undefined variables, unsafe patterns, unused code).
- Preserve practical style where needed for readability instead of forcing strict mechanical uniformity.

## 2) JavaScript / UI Formatting

- Prefer clear section comments for complex files (`State`, `Handlers`, `Rendering`, `Events`).
- Keep component/view logic grouped and readable.
- Break dense function calls, object literals, and complex ternaries into multiline blocks.
- Keep long utility class strings inline unless splitting clearly improves readability.
- Mixed practical style (semicolons/no-semicolons) is acceptable if a file stays internally coherent.

## 3) Tailwind / Utility-Class Conventions

When writing utility-first class stacks, prefer this rough order:

1. layout/position
2. spacing
3. typography
4. color/background
5. border/shadow/effects
6. transitions/interactions

Prefer subtle gradients, layered opacity, and small interaction states for polished UI feel.

## 4) CSS Organization

- Group styles by logical sections with concise comments.
- Favor reusable semantic utility classes for repeated motifs.
- If Tailwind layers are used, organize custom CSS into `@layer base` and `@layer utilities`.

## 5) Markdown Output Formatting (Critical)

Generated explanatory content should prefer:

- `##` section headings
- `**bold**` for key terms
- `>` blockquotes for critical warnings/disclaimers
- bullet lists for steps/recommendations

Keep sections concise (typically 2-4 sentences) and normalize ad-hoc titles into valid markdown headings.

## 6) Documentation Style

- Use emoji-based section headers in docs when appropriate.
- Use strong emphasis for key concepts.
- Prefer numbered setup/deployment steps with sub-bullets.
- Include command examples in fenced code blocks.

## 7) Tone and Safety

- Keep UI/documentation copy concise, reassuring, and action-oriented.
- In health/risk contexts, explicitly include safety disclaimers (for example: not a substitute for professional care).

## Workflow Commands

```bash
# Install JS tooling once
npm install

# Auto-format frontend and docs
npm run format

# Check formatting in CI/local
npm run format:check

# Run correctness-focused frontend lint
npm run lint:js
```
