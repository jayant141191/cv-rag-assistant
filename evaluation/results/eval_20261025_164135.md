# CV RAG — Evaluation report

- Run: 2026-10-25T12:41:35+00:00
- Chat model: `gpt-4o-mini` · Judge: `gpt-4o` · Embeddings: `text-embedding-3-large`
- Examples: 15

## Overall

| Metric | Score |
|---|---|
| retrieval_hit | 0.86 |
| retrieval_mrr | 0.76 |
| candidate_recall | 0.83 |
| candidate_precision | 0.76 |
| context_keyword_recall | 0.81 |
| answer_keyword_recall | 0.76 |
| correctness | 4.20 |
| faithfulness | 4.33 |
| relevance | 4.87 |
| latency_s | 4.84 |

## By category

| Category | n | retrieval_hit | retrieval_mrr | candidate_recall | answer_keyword_recall | correctness | faithfulness | relevance |
|---|---|---|---|---|---|---|---|---|
| comparison | 2 | 1.00 | 1.00 | 0.83 | 0.83 | 1 | 1 | 4 |
| multi_candidate | 1 | 1.00 | 1.00 | 1.00 | 0.50 | 3 | 3 | 5 |
| off_topic | 1 | – | – | – | – | 5 | 5 | 5 |
| search | 3 | 0.33 | 0.33 | 0.33 | 0.33 | 4.33 | 5 | 5 |
| single_fact | 6 | 1.00 | 0.76 | 1.00 | 1.00 | 5 | 5 | 5 |
| unanswerable | 2 | 1.00 | 1.00 | 1.00 | – | 5 | 5 | 5 |

## Per question

| ID | Category | Hit | MRR | Correct | Faithful | Judge note |
|---|---|---|---|---|---|---|
| q01 | single_fact | 1.00 | 1.00 | 5 | 5 | The answer correctly states that Priya Raman completed her bachelor's degree in Computer Science at the National Institute of Technology Tri |
| q02 | single_fact | 1.00 | 0.25 | 5 | 5 | The answer correctly identifies Daniel Okafor's current job and employer as a Machine Learning Engineer at Lumen Health, matching the refere |
| q03 | single_fact | 1.00 | 1.00 | 5 | 5 | The answer correctly lists the certifications held by Sofia Martinez as per the reference answer and the retrieved context. It is faithful t |
| q04 | single_fact | 1.00 | 1.00 | 5 | 5 | The answer accurately reflects the information provided in the retrieved context, matching the reference answer in terms of the size of the  |
| q05 | single_fact | 1.00 | 0.33 | 5 | 5 | The answer correctly states the F1 improvement from 0.81 to 0.89, which matches the reference answer and is supported by the retrieved conte |
| q06 | single_fact | 1.00 | 1.00 | 5 | 5 | The answer correctly lists all the languages Sofia Martinez speaks as per the reference answer and the retrieved context. It is fully faithf |
| q07 | search | 0.00 | 0.00 | 3 | 5 | The answer correctly identifies that Jayant Sonawane has experience with a RAG system, as supported by the context. However, it misses the s |
| q08 | search | 1.00 | 1.00 | 5 | 5 | The answer correctly identifies Priya Raman as having worked with CDC, specifically mentioning the introduction of CDC replication from Auro |
| q09 | search | 0.00 | 0.00 | 5 | 5 | The answer correctly abstains from providing information not available in the context, as there is no retrieved context to support any claim |
| q10 | multi_candidate | 1.00 | 1.00 | 3 | 3 | The answer correctly identifies Priya Raman's management experience, which is supported by the context. However, it fails to mention Sofia M |
| q11 | comparison | 1.00 | 1.00 | 1 | 1 | The answer incorrectly includes Jayant Sonawane, who is not mentioned in the reference answer or the retrieved context. The reference answer |
| q12 | comparison | 1.00 | 1.00 | 1 | 1 | The answer incorrectly concludes that Jayant Sonawane is the better fit for the AWS streaming data platform role, despite the reference answ |
| q13 | unanswerable | 1.00 | 1.00 | 5 | 5 | The answer correctly states that the expected salary is not mentioned, aligning with the reference answer. It is faithful to the lack of con |
| q14 | unanswerable | 1.00 | 1.00 | 5 | 5 | The answer correctly states that there is no information available in the indexed CVs regarding Sofia Martinez's experience with Kubernetes, |
| q15 | off_topic | – | – | 5 | 5 | The answer correctly redirects the question to CV-related topics, as the question about the capital of France is off-topic. This matches the |
