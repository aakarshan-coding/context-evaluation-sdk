# Label audit on a real bot

This audit checks whether contexteval's failure labels are right on a real RAG bot (`bot.py`) with a real model. The bot runs on `gpt-4.1-mini` with `text-embedding-3-small` embeddings and calls the model at its default temperature. contexteval graded with `gpt-4.1-mini`. Every label below was checked by hand against its stored trace. That means the retrieved chunks, the chunks in the prompt, the answer, and the replay answers. The tools are in `audit/` and are described in `README.md`.

## Runs

The test set has 75 cases. `build-tests --per-passage 2` generated 47 from the corpus, with 1 rejected for non-verbatim evidence. The other 28 are the hand-written `customer_questions.jsonl`.

| Run | Setting | Cases | Passed | Labels |
|---|---|---|---|---|
| 1 | baseline, synthetic cases only | 47 | 45 | missing 2 |
| 2 | baseline | 75 | 70 | missing 5 |
| 3 | baseline (experiment 1) | 75 | 70 | missing 5 |
| 4 | top_k=8 | 75 | 74 | missing 1 |
| 5 | top_k=8 rerank=lexical | 75 | 73 | missing 1, ignored 1 |
| 6 | baseline (experiment 2) | 75 | 70 | missing 5 |
| 7 | top_k=15 order=reverse | 75 | 74 | flaky 1 |
| 8 | top_k=15 order=ends | 75 | 72 | ignored 2, irrelevant 1 |

In run 2 the generated cases passed 45 of 47 (96%) and the customer questions passed 25 of 28 (89%). Generated questions reuse the wording of their source passage, which makes retrieval easier than real customer phrasing.

Experiment tables, as printed:

```text
setting                 accuracy  passed  mean input tokens  mean latency ms  total cost
baseline                93%       70/75   404                871.3            $0.0159
top_k=8                 99%       74/75   760                881.7            $0.0266
top_k=8 rerank=lexical  97%       73/75   760                912.9            $0.0267

baseline                93%       70/75   404                897.8            $0.0158
top_k=15 order=reverse  99%       74/75   1376               857.7            $0.0453
top_k=15 order=ends     96%       72/75   1376               960.9            $0.0454
```

## Every label

Chunk positions are 0-based, as contexteval prints them.

| Run, trace | Question | contexteval label | Verdict | Reason |
|---|---|---|---|---|
| 1, 22 | How long does it take to fully charge the Metro-E battery? | missing (not_retrieved) | label correct | The charge time is in `metro-e.md#1`, which is not in the top 4. The bot said it did not know. Injecting the evidence gave 4.5 hours. |
| 2, 69 / 3, 144 / 6, 369 | (same) | missing (not_retrieved) | label correct | Same retrieval, same "don't know" answer. top_k=8 passes it. |
| 1, 39 | What type of brakes does the Trailhawk 29 Hardtail have? | missing (not_retrieved) | label correct | The spec list `trailhawk-29.md#1` is not in the top 4. Three of the four chunks are Trailhawk Pro chunks. |
| 2, 86 / 3, 161 / 6, 386 | (same) | missing (not_retrieved) | label correct | Same as above. |
| 2, 104 / 3, 179 / 6, 404 | How heavy is the Trailhawk Pro? | missing (not_retrieved) | label correct | The weight is in the Pro spec list `trailhawk-pro.md#1`, which ranks below the intro, the ownership notes, the Trailhawk 29 intro, and the FAQ. The bot said it did not know. |
| 2, 113 / 3, 188 / 6, 413 | When do Ridgeline Rewards certificates expire? | missing (not_retrieved) | label correct | `loyalty-program.md#0` was retrieved, but the expiry sentence is in `#1`, which was not. |
| 2, 116 / 3, 191 / 6, 416 | What's the most weight the Metro-E can carry, including me? | missing (not_retrieved) | label correct | `metro-e.md#1` is not in the top 4. The bot said it did not know. |
| 4, 266 | (same), top_k=8 | missing (not_retrieved) | label correct | `metro-e.md#1` is still not in the top 8. Now the bot answers wrongly with the Cargo's 200 kg payload. |
| 5, 341 | (same), top_k=8 rerank=lexical | missing (not_retrieved) | label correct | Same 8 chunks, reordered. Same wrong 200 kg answer. |
| 5, 311 | Trailhawk 29 brakes, top_k=8 rerank=lexical | ignored, gold at chunk 5 | label correct | Lexical rerank lifted the Cargo spec list, which contains "Brakes", to chunk 3. The bot answered with the Cargo's four-piston brakes. 5 extra replays gave baseline 0/5, gold to top 5/5, only gold 5/5. |
| 7, 461 | Trailhawk 29 brakes, top_k=15 order=reverse | flaky | grading wrong | The answer said "hydraulic disc brakes with 180 mm rotors front and rear". That is true of this bike but leaves out "Shimano MT410", and the grader failed it. If the grade stands, the label is still weak: 5 extra baseline reruns all failed, and gold to top passed 5/5, so ignored fits better. |
| 8, 519 | How long does it take to fully charge the Metro-E battery?, top_k=15 order=ends | ignored, gold at chunk 11 | label correct | The bot took the Cargo's 6 hours from chunk 10 and joined it to the Metro-E's 4 A charger. 5 extra replays gave baseline 0/5, gold to top 5/5, only gold 5/5. |
| 8, 535 | What is the price of the Trailhawk 29 Hardtail?, top_k=15 order=ends | ignored, gold at chunk 13 | label wrong, should be irrelevant | The bot answered $4,199, the Pro's price. That price is in the Pro spec list at chunk 1, right after the Trailhawk 29 intro at chunk 0. The one gold to top replay passed, but 5 extra replays gave gold to top 2/5 and only gold 5/5. The ignored label came from a minority outcome. |
| 8, 554 | How heavy is the Trailhawk Pro?, top_k=15 order=ends | irrelevant | label correct | The bot gave the Cargo's 35 kg "with one battery" for the Pro. Gold to top failed, and the bot said the weight was not given, because the spec list has no product name. Only gold passed. 5 extra replays gave gold to top 0/5 and only gold 5/5. |

## Passing cases

All 70 passing answers in run 2 were read against their expected answers (`audit/passes.py`). 50 passed on the substring rule and 20 through the LLM grader. Six were checked in depth against their traces:

| Trace | Question | Verdict | Reason |
|---|---|---|---|
| 100 | What time does the Boulder service department close on Saturdays? | true pass | The bot combined "closes one hour before the store" with Saturday's 6 pm close and said 5 pm. |
| 103 | I'm 5'11". What size Trailhawk 29 should I get? | true pass | Large, with the 5'10" to 6'1" range quoted. |
| 119 | How much credit would I get for trading in my 6-year-old road bike? | true pass | Up to 40% of the used-resale appraisal, as store credit. |
| 88 | What type of suspension does the Trailhawk Pro have? | true pass, weak case | The generated expected answer is "carbon full-suspension", which mixes in the frame material. The answer "full suspension" is right. |
| 102 | Can I get my Metro-E motor fixed at the Fort Collins shop? | true pass | Says no and that the repair goes to Boulder. It leaves out the two extra business days. |
| 121 | Can I trade in my 7-year-old e-bike? | false pass, borderline | The answer opens "Yes, you can trade in your 7-year-old e-bike" and then says the bike does not meet the 5-year limit. The LLM grader passed it. |

`audit/polarity.py` looked at all 548 passing traces in every run. Trace 121 is the only pass whose answer opens with the opposite yes or no. `audit/goldcheck.py` found that every passing trace in all 8 runs had a gold chunk in the prompt. Every not_retrieved trace failed. So the gold locator never contradicted the outcome.

## Summary

Each of the 24 stored labels was audited. 22 are correct. 1 label is wrong (trace 535, ignored where irrelevant is better supported). 1 failure is a grading error (trace 461, a correct but less specific answer graded as wrong). Label precision is 22 of 23 on correctly graded failures. Many labels repeat the same question under the same setting. Counting each question and setting once gives 12 distinct labels: 10 correct, 1 wrong, and 1 grading error. The passing cases have 1 borderline false pass out of 70 (trace 121).

The 19 missing labels are all correct, and they are robust. contexteval decides missing from where the gold chunk sits, not from a replay, and the inject replay confirmed every one. The experiment agrees with them. top_k=8 fixed 4 of the 5 baseline missing cases. The fifth never ranks in the top 8.

## Systematic findings

**Single replays at app temperature.** contexteval decides flaky, ignored, and irrelevant from one replay per step, and the bot samples at default temperature. Five replays per step changed the reading of 2 of the 5 in-prompt labels. Trace 461's flaky came from a rerun that passed once, when 5 more reruns failed. Trace 535's ignored came from a gold to top replay that passes 2 times in 5. The other three labels were 5 of 5 or 0 of 5 on every step. This matches the documented behavior ("If it passes now, the label is `flaky`"), so it is a design limit, not a code bug. The fix would be several replays per step with a majority rule, at 3 to 5 times the replay cost. That is a product decision, so this audit does not make it.

**The labels don't name the real cause.** All five in-prompt failures share one cause in the bot. The chunker cuts each product page's heading away from its spec list, so a spec-list chunk carries no product name. With more chunks in the prompt, the model takes specs from the wrong product: the Cargo's brakes, weight, and charge time, and the Pro's price. Whether that shows as ignored or irrelevant depends on which chunk sits next to the product intro. The README maps ignored to reordering and irrelevant to reranking. Here both only help partly. The fix that addresses the cause is to put the document title in each chunk. contexteval surfaced the failures and the replay answers make the cause visible, but the label text does not point there.

**Lexical rerank made this bot worse.** It promotes chunks that share a keyword with the question. For "What type of brakes does the Trailhawk 29 have?" it lifted the Cargo spec list, which contains "Brakes". Accuracy fell from 99% to 97% against top_k=8 alone.

**The experiment's app cost line underestimates larger top_k.** The line reads "~404 input tokens each in the last run". That is the baseline's figure, so the top_k=15 rows actually used 1,376 tokens, 3.4 times more. The line says where the number comes from, so it does not misstate anything, but it understates the cost of the settings being tested.

**Positions are 0-based.** "gold was at chunk 5" means the sixth chunk. The dashboard numbers chunks the same way, so the tool is consistent with itself. A reader of the CLI line alone may count from 1.

No code bug in contexteval turned up, so the audit changes no code. Every wrong reading traces to the single-sample replay design or to grading strictness. It does not trace to the gold locator, the replay transforms, or the ladder order, which all behaved as documented on every trace.

## Spend

`audit/tally.py` wrapped `openai`'s `SyncAPIClient.request` and summed reported token usage at $0.40 and $1.60 per million input and output tokens for gpt-4.1-mini and $0.02 per million for text-embedding-3-small.

| Command | App calls | SDK calls | Embedding calls | Cost |
|---|---|---|---|---|
| build-tests | 0 | 24 | 0 | $0.0086 |
| run (2 runs) | 122 | 33 | 123 | $0.0274 |
| diagnose (8 runs) | 48 | 24 | 48 | $0.0155 |
| experiment (2, 6 rows) | 450 | 135 | 450 | $0.1826 |
| stability replays | 90 | 38 | 90 | $0.0400 |
| total | 710 | 254 | 711 | $0.2740 |
