Please assign appropriate heading levels (# to ######) to the following Markdown headings based on their hierarchical relationships.

## Rules

1. **The first heading** is usually `#` (h1).
2. **Infer level from numbering**: If a heading contains numbering (e.g., 1, 1.1, 1.1.1), determine the heading level based on the numbering hierarchy:
   - `1` → `##` (h2)
   - `1.1` → `###` (h3)
   - `1.1.1` → `####` (h4)
   - `1.1.1.1` → `#####` (h5)
   - and so on.
3. **Infer from context**: If a heading has no numbering, infer its level based on context and indentation.
4. **Keep content unchanged**: Only adjust the number of `#` symbols; keep the heading text content unchanged.
5. **Batch connection**: If the input contains a `---` separator, the preceding part contains previous headings (for hierarchical reference), and the following part contains headings to be formatted. Based on the numbering pattern of the preceding part and its connection to the following part, assign appropriate levels to the following part. **Only output** the formatted result for the following part.
   - Common hierarchy: `Chapter X`(#) → `Section X`(##) → `一、二、`(###) → `1. 2.`(####)
   - Ensure hierarchical continuity at batch boundaries. For example, if the preceding part ends with `#### 5. xxx`, then `六、yyy` in the following part should be `####`.

## Output Requirements

- Only output the formatted headings, one per line.
- Maintain the original order.
- Do not add any other content (e.g., explanations, comments).

## Example

### Input:
```
# Title
# 1 title1
# 1.1 title1.1
# 1.1.1 title1.1.1
```

### Output:
```
# Title
## 1 title1
### 1.1 title1.1
#### 1.1.1 title1.1.1
```

---

Now, please format the following headings:

{content}