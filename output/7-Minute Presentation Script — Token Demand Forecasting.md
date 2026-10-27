# 7-Minute Presentation Script — Token Demand Forecasting

Oct 23, 2026 · @HYsynergy

## Timing guide

The script runs about 1,000 words, which is 7 minutes at a steady 145 words per minute. It follows the current deck order and shows one Hypothesis slide and one Pipeline slide, so skip the duplicates.

| Slide | Section | Time | Ends at |
| --- | --- | --- | --- |
| 1–2 | Title, Contents | 15 s | 0:15 |
| 3 | Background & Motivation | 40 s | 0:55 |
| 4 | Related Work | 45 s | 1:40 |
| 5 | Research Question | 20 s | 2:00 |
| 6 | Hypothesis | 45 s | 2:45 |
| 8–10 | Dataset, Data Fields, How We Use It | 55 s | 3:40 |
| 11–13 | Forecast Target, Evaluation, Pipeline | 75 s | 4:55 |
| 15–17 | EDA ① ② ③ | 100 s | 6:35 |
| 18–20 | Workload, Beyond, Questions | 25 s | 7:00 |

If you run long, shorten Related Work to one line per paper. Data Fields and Pipeline can also be a single sentence each.

## Opening (0:00–0:55)

**Slide 1 · Title** (0:00–0:10)

Hello everyone. We are Team 4: Lee Junsang, Seo Dael, Roh Jonghyeon, and Lim Jihun. Our proposal is about forecasting LLM token demand using conversation state.

**Slide 2 · Contents** (0:10–0:15)

We'll start with motivation and related work. Then we'll cover our question, the data, our method, early EDA results, and our plan.

**Slide 3 · Background & Motivation** (0:15–0:55)

Demand for LLM services is growing fast, and that creates a capacity problem. With too few servers, requests get delayed or fail. With too many, power is wasted. So capacity has to follow demand.

The catch is that a new server takes minutes to start. Reacting after demand arrives is too late. We need to know minutes ahead.

And it's not just the number of requests that matters. The load depends on how long the inputs and outputs are. So we forecast input and output tokens. That leads to our motivating question: can we know token demand in advance?

## Problem framing (0:55–2:45)

**Slide 4 · Related Work** (0:55–1:40)

Four recent works come close to this.

BurstGPT, from KDD 2025, released the real logs we use. It forecasts request and token counts with XGBoost, but only from past aggregates.

DynamoLLM tunes server count and GPU settings to the load and saves 53 percent of energy. But it predicts the lengths of requests that have already arrived.

PreServe forecasts total token demand and per-request length, but it ignores how requests link into conversations.

SageServe, from Microsoft, uses ARIMA to scale ahead of a roughly ten-minute model redeployment. But it works on hourly aggregates, with no input–output split.

So here is the common gap: none of them forecasts the next demand from the state of ongoing conversations.

**Slide 5 · Research Question** (1:40–2:00)

That gives us our research question. Does adding information about ongoing conversations to demand history forecast next-ten-minute token demand more accurately? The intuition is simple: someone in an active conversation is likely to send another message soon.

**Slide 6 · Hypothesis** (2:00–2:45)

We test this with three nested models. They share the same forecast origins, the same algorithm, and the same tuning budget.

Model A is our baseline, in the style of BurstGPT. It uses lags and rolling statistics of request count, input tokens and completed output tokens, plus request length and time of day.

Model B adds the active conversation count: how many unique conversations sent a request in the last 5, 15 and 30 minutes.

Model C also adds a conversation state summary. For each conversation active in the last 30 minutes, it uses the time since its last request, its request count, its age, and its last input length and how that length changed.

H1 says B has lower error than A. H2 says C has lower error than B.

## Data (2:45–3:40)

**Slide 8 · Dataset** (2:45–3:15)

Our data is BurstGPT version 2.0. These are real GPT request logs from a regional Azure OpenAI service used by more than 3,000 people.

The logs cover 110 days and about 5.3 million requests. Of those, 233,617 are conversation-log requests from 55,920 conversations, and 36,086 of those conversations have two or more requests.

Every request has a timestamp, token counts, and a conversation ID. That lets us model demand over time and test whether conversation activity helps.

**Slide 9 · Data Fields** (3:15–3:25)

The key fields are the timestamp and the session ID, which links requests in one conversation. We also use input and output tokens, and the elapsed time, which tells us when each response completes.

**Slide 10 · How We Use It** (3:25–3:40)

We keep the conversation logs and aggregate them into one-minute series. We build conversation features from past requests by session ID, and we forecast every minute. Crucially, we only use what is known at the forecast origin. Output tokens count only after a response finishes.

## Method (3:40–4:55)

**Slide 11 · Forecast Target** (3:40–4:10)

At each minute t, we forecast two sums for the requests arriving in the next ten minutes. The first is input demand, I of t. The second is output demand, O of t, which is the final output tokens of those same requests.

Take a 10:00 forecast origin. The input label is final at 10:10. The output label is final only once all of those responses finish, around 10:12 in this example. Because we forecast every minute, adjacent windows overlap by nine minutes.

**Slide 12 · Evaluation** (4:10–4:40)

We split the data chronologically: 60 percent for training, 20 for validation, and 20 for testing. A, B and C share origins, labels, algorithm and tuning budget, so any difference comes from the features alone.

MAE is our primary metric and RMSE is secondary. We skip MAPE because some targets are zero. We report B minus A, C minus B, and C minus A, separately for input and output.

We also train only on labels that are finalized by training time, and we account for the error dependence caused by overlapping windows.

**Slide 13 · Pipeline Overview** (4:40–4:55)

Putting it together: the conversation logs become one-minute demand history plus conversation state. These feed separate XGBoost regressors for input and output, for each of A, B and C. The result is a next-ten-minute forecast and a measure of how much conversation state adds.

## EDA (4:55–6:35)

**Slide 15 · EDA ① Target Demand** (4:55–5:25)

Now for our early EDA, starting with the target itself. In the daily view, short demand peaks appear next to quiet periods. Across the whole trend, the scale of demand also shifts over the observation period. So recent demand history and time of day go into every model as common inputs.

**Slide 16 · EDA ② Conversations** (5:25–6:00)

Second, conversations. 64.5 percent of conversations have two or more requests, and those repeated conversations hold 91.5 percent of all requests. The median gap before a follow-up request is 131 seconds. Median input length also changes with request order. So recent conversation activity and state are natural candidate inputs.

This also set our horizon. The share of follow-ups we capture rises from 66.1 to 76.7 percent when we go from 5 to 10 minutes. That's a gain of 10.6 points. Going to 15 minutes adds only 4.9 more. So ten minutes captures most follow-ups.

**Slide 17 · EDA ③ Correlations** (6:00–6:35)

Third, correlations. Request count in the last 5 minutes has Spearman correlations of 0.822 and 0.832 with next-ten-minute input and output tokens. Active conversations in the last 5 minutes score slightly higher, at 0.825 and 0.841. Both are strong candidates.

But the two are correlated with each other at 0.978, so a simple count may add little on its own. That's exactly why we compare B against A, and C against B: to see whether conversation state adds information beyond simple counts.

## Closing (6:35–7:00)

**Slide 18 · Workload Distribution** (6:35–6:45)

Here is how we split the work. Lee Junsang builds the models, Seo Dael designs the features and labels, Roh Jonghyeon handles preprocessing and EDA, and Lim Jihun leads evaluation. We share the experiment design and the leakage checks.

**Slide 19 · Beyond This Project** (6:45–6:57)

This project asks one question: does conversation state reduce forecast error? Actually scaling servers is out of scope. But DynamoLLM showed that tuning to load can save 53 percent of energy, and doing that proactively requires knowing demand first. So if our forecasts work, the next question is whether proactive scaling cuts power and latency.

**Slide 20 · Questions** (6:57–7:00)

Thank you. We're happy to take your questions.
