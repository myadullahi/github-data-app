# RAG benchmark (for reranking / Step 2)

Use this as a **reference benchmark** when refining the reranking approach. No automated action; for manual testing.

---

## Benchmark question chain 1

Run this sequence and compare behavior as you improve retrieval/reranking:

1. **Question:**  
   *"give me top-3 questions asked in data engineering interview in spark"*

2. **Follow-up:**  
   *"what are the answers of these?"*

3. **Follow-up:**  
   *"what is the source repo name and link?"*

---

## Benchmark question chain 2

Second reference flow (also use when testing reranking):

1. **Question:**  
   *"give me top-3 data engineering questions on spark"*

2. **Follow-up:**  
   *"give me the answers"*

**Bad example (current behavior to improve):** The system cites repos such as `absognety/Interview-Process-Coding-Questions`, `andkret/Cookbook`, or `josephmachado/data_engineering_best_practices` — which often list only questions or homework formats, not full Q&A. The model then says "the context does not contain answers" or points users to docs. That is a **poor outcome**.

**Better source (what we want to surface):** The repo **OBenner/data-engineering-interview-questions** contains actual Q&A content, e.g. in `content/spark.md`. As we improve reranking, we want to prefer chunks from such content so the user gets real answers and useful repo links.

**Good example (target quality):** For *"give me the top-3 data engineering questions on spark"*, a good response includes both the three questions and substantive answers (e.g. what is RDD, Spark vs MapReduce, Spark and Hadoop), with **specific GitHub repo URLs** (not [1]/[2]). Example shape:

- 1. What is RDD and how do you use it? — short answer + source URL  
- 2. What is the difference between Spark and MapReduce? — short answer + source URL  
- 3. How does Spark use data from Hadoop? — short answer + source URL  
- Sources: multiple repos listed with links  

**Source diversity (avoid single-repo bias):** If the model cites only one repo (e.g. andkret/Cookbook) for all answers even when others (e.g. OBenner/data-engineering-interview-questions) are in the retrieved set, that is a **bias** we want to reduce. Rerank uses a per-repo cap so the top chunks are spread across multiple repos when possible.

---

## What we're improving (not "expected answers")

- **Current / bad examples:**
  - **Chain 1:** The system sometimes surfaces a repo (e.g. a list that says "answers are for losers" / encourages self-study) and gives a reply based on that single perspective.
  - **Chain 2:** The system surfaces question-only or lightweight repos and replies that "context does not contain answers," even when better sources (e.g. OBenner/data-engineering-interview-questions with `content/spark.md`) exist in the index and do contain answers.
- **Goal:** As we explore reranking (and dense + sparse + rerank), we want to surface **better sources** that actually contain substantive answers (e.g. explanations of repartition vs coalesce, what is shuffle, how to optimize Spark queries). Better ranking should prefer those over generic or meta "no answers" content and over question-only lists.
- Use these benchmarks to check that answers and cited sources **improve** as we refine the pipeline (e.g. Step 2 reranking). There is no single "expected" repo or link; the bar is "better, more relevant sources and answers."

---

## Requirement: specific repo links (not [1], [2])

- **Output must not** say only "source: [1]" or "source: [2]".
- **Output must** give the **specific GitHub repo URL** for any cited source (e.g. `https://github.com/owner/repo`).
- The app is updated so that:
  - The model is prompted to cite using full GitHub URLs.
  - The API returns a `sources` list with `repo` and `url` for each retrieved source.
  - The chat UI shows these as clickable links under the answer.

Use this benchmark to validate that behavior improves before/after reranking and other retrieval changes.
