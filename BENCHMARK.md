# RAG benchmark (for reranking / Step 2)

Use this as a **reference benchmark** when refining the reranking approach. No automated action; for manual testing.

---

## Five tests to run (before any code changes)

Use these 5 tests as the core checklist when testing the RAG app (manual or later automated):

| # | Test | Input (user asks) | Pass criteria |
|---|------|-------------------|----------------|
| **1** | **Spark Q&A chain** | (1) "give me top-3 questions asked in data engineering interview in spark" → (2) "what are the answers of these?" → (3) "what is the source repo name and link?" | Response has 3 questions, then substantive answers (not "context has no answers"), and each answer/source has a **specific GitHub URL** (no [1]/[2]). Final reply gives repo name and link. |
| **2** | **Follow-up answers (no refusal)** | (1) "top-3 overall data engineering interview questions" → (2) "can you also give the answers to those questions" | Second response **provides the actual answers** from context (e.g. 4 V's, batch vs streaming, ACID or similar), with source URLs. Does **not** say "I'm unable to provide answers" or only point to repos. |
| **3** | **Citations: full URLs + per-answer** | Any multi-part answer (e.g. top-3 questions with answers) | No generic "source: [1]" or "source: [2]". Every cited source is a **full GitHub URL** (e.g. https://github.com/owner/repo). When listing several Q&As, **each answer has its source directly after it**, not only a single sources section at the end. |
| **4** | **Source diversity** | "give me top-3 data engineering questions on spark" (or similar list + answers) | Cited sources include **more than one repo** when the index has multiple relevant repos (e.g. not only andkret/Cookbook). Reranked chunks (logs or API) show multiple repos in the top set. |
| **5** | **RAG-only (no general knowledge)** | (a) Ask something with a **specific phrase** that exists in only one indexed repo — answer should cite that repo and reflect that phrasing. (b) Ask something **not** in the indexed repos — response should say answer is not in context / cannot find it, **not** give a full answer from elsewhere. | Answers are grounded in retrieved context; when context is missing, the model says so instead of answering from general knowledge or "online" sources. |

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

## Benchmark question chain 3 (follow-up answers)

1. **Question:**  
   *"top-3 overall data engineering interview questions"*

2. **Follow-up:**  
   *"can you also give the answers to those questions"*

**Bad example (not acceptable):** The model first lists three questions (e.g. 4 V's of Big Data, batch vs streaming, ACID properties) with sources, but when the user asks for the **answers**, it replies with "I'm unable to provide answers" or "you can find the information in the following repositories" and only points to links. That is a **poor outcome**: the indexed context often **does** contain Q&A content (e.g. [OBenner data-modeling.md](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/data-modeling.md) and similar files with question + answer per section). The system must **use that context** to provide the actual answers, not refuse.

**Expected behavior:** When the user asks for answers to previously listed questions, the assistant should pull the answers from the provided context when they exist (e.g. short explanations for 4 V's, batch vs streaming, ACID), cite the source URL, and only say "not in context" when the context truly does not contain that answer.

---

## What we're improving (not "expected answers")

- **Current / bad examples:**
  - **Chain 1:** The system sometimes surfaces a repo (e.g. a list that says "answers are for losers" / encourages self-study) and gives a reply based on that single perspective.
  - **Chain 2:** The system surfaces question-only or lightweight repos and replies that "context does not contain answers," even when better sources (e.g. OBenner/data-engineering-interview-questions with `content/spark.md`) exist in the index and do contain answers.
  - **Chain 3:** The system lists questions correctly but when the user asks for **answers** to those questions, it refuses ("I'm unable to provide answers") or only points to repos instead of extracting and providing the answers from the context (e.g. OBenner `content/data-modeling.md`, `content/spark.md`, which have question + answer per section).
- **Goal:** As we explore reranking (and dense + sparse + rerank), we want to surface **better sources** that actually contain substantive answers (e.g. explanations of repartition vs coalesce, what is shuffle, how to optimize Spark queries). Better ranking should prefer those over generic or meta "no answers" content and over question-only lists.
- Use these benchmarks to check that answers and cited sources **improve** as we refine the pipeline (e.g. Step 2 reranking). There is no single "expected" repo or link; the bar is "better, more relevant sources and answers."

---

## Requirement: specific repo links (not [1], [2])

- **Output must not** say only "source: [1]" or "source: [2]".
- **Output must** give the **specific GitHub repo URL** for any cited source (e.g. `https://github.com/owner/repo`).
- **Per-answer citation:** When listing multiple Q&As (e.g. top-3 interview questions with answers), put the source **directly with each answer** (e.g. "Source: https://github.com/...") right after that answer, not only in a separate "sources" section at the end.
- The app is updated so that:
  - The model is prompted to cite using full GitHub URLs.
  - The API returns a `sources` list with `repo` and `url` for each retrieved source.
  - The chat UI shows these as clickable links under the answer.

Use this benchmark to validate that behavior improves before/after reranking and other retrieval changes.

---

## Milestone-3 validation: RAG-only answers

When doing milestone-3 testing, **one of the tests must check that results come only from the RAG context** (the indexed GitHub repos), not from the model’s general knowledge or other online sources.

**How to check:**

- Ask a question that has a **specific wording or detail** that appears in only one indexed repo (e.g. a phrase from OBenner’s data-modeling.md or a repo-specific definition). Confirm the answer matches that wording or cites that repo; if the model gives a different, “generic” answer without a cited chunk, it may be using non-RAG knowledge.
- For a topic that is **not** in the indexed repos (or remove that repo from the index for the test), the assistant should say the answer is not in the context or that it cannot find it—**not** give a full answer from general knowledge.
- Optionally: compare the model’s answer to the **reranked_chunks** returned by the API; the answer should be supportable from those chunks (wording, facts, or structure traceable to the context).

**Pass criteria:** Answers are grounded in the retrieved context; when context is missing or irrelevant, the model says so instead of answering from elsewhere.
