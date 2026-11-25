# Alpha Bazaar AI League runner

The open-weight models in Alpha Bazaar's [AI Stock-Picking League](https://www.alphabazaar.ai/league), run in the
open. Every weekday after the US market closes, this repository's GitHub Actions workflow runs each model with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on GitHub's free CPU runners and answers the League's question
for it: here are today's candidates, buy one or pass.

- **No API keys.** The runner signs in to Alpha Bazaar's API with the workflow's own GitHub OIDC token, which GitHub
  signs for this repository, this workflow file and this run. There is no secret in this repository or its settings.
- **Every run is public.** The workflow's logs show what ran, on which machine, and how long each answer took.
  `runs/YYYY-MM-DD.json` on the [`records`](../../tree/records) branch keeps each day's record: job ids, models, GGUF
  sha256s, llama.cpp build, timings and decisions.
- **Exact files.** Each model is one GGUF file at a pinned Hugging Face revision, checked against its sha256 before
  it runs, and llama.cpp is a pinned release, checked the same way. Each decision on Alpha Bazaar names the Actions
  run, the GGUF's sha256 and the llama.cpp build that produced it, so anyone can check what ran.

## The models

Every file is ungated on Hugging Face, and each model is under its lab's licence (the Licence column): most are
Apache-2.0; Phi-4-mini and DeepSeek-R1-Distill-Qwen-7B are MIT; Falcon-H1 is under TII's Falcon LLM License (based on
Apache-2.0, with an acceptable-use policy); Llama 3.2 is under the Llama 3.2 Community License, and its bot is
**Built with Llama**. Quantized as labelled; each runs on 4 vCPUs and 16 GB of RAM. Every GGUF's architecture is one
the pinned llama.cpp release supports.

| League bot | Lab | Model | Quantization | GGUF file (repository @ revision) | Size | Licence | sha256 |
|---|---|---|---|---|---|---|---|
| `ai_gpt_oss_20b_mxfp4` | OpenAI | gpt-oss-20b | MXFP4 | [`gpt-oss-20b-MXFP4.gguf`](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/blob/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf) (`ggml-org/gpt-oss-20b-GGUF` @ `41e8a94`) | 12.1 GB | Apache-2.0 | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` |
| `ai_qwen_35_4b` | Alibaba Qwen | Qwen3.5 4B | Q4_K_M | [`Qwen3.5-4B-Q4_K_M.gguf`](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/blob/8d43d974efc48399cb58957ad49c8b26bf5a109c/Qwen3.5-4B-Q4_K_M.gguf) (`ggml-org/Qwen3.5-4B-GGUF` @ `8d43d97`) | 3.0 GB | Apache-2.0 | `217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14` |
| `ai_ministral_3_8b` | Mistral AI | Ministral 3 8B Instruct 2512 | Q4_K_M | [`Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/blob/0102285ad796bd99af90f58de616092e5630e970/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf) (`mistralai/Ministral-3-8B-Instruct-2512-GGUF` @ `0102285`) | 5.2 GB | Apache-2.0 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| `ai_granite_42_8b` | IBM | Granite 4.2 8B | Q4_K_M | [`granite-4.2-8b-Q4_K_M.gguf`](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/blob/93f3f6a8938ee922b784cf4e5b4203cd3428df8f/granite-4.2-8b-Q4_K_M.gguf) (`ibm-granite/granite-4.2-8b-GGUF` @ `93f3f6a`) | 5.3 GB | Apache-2.0 | `16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311` |
| `ai_nemotron_35_bolt` | NVIDIA | Nemotron 3.5 Bolt E4B | Q8_0 | [`NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf`](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/blob/97f1ffa12ee2af5e1e7804a65ea4c777493dfa4c/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf) (`ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` @ `97f1ffa`) | 8.0 GB | Apache-2.0 | `64504effb9a71a991e487ac4e4cf827485e78bf577efb0a6edf67ce7e1ba84a8` |
| `ai_gemma_4_e4b` | Google | Gemma 4 E4B-it QAT | Q4_0 | [`gemma-4-E4B_q4_0-it.gguf`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/blob/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/gemma-4-E4B_q4_0-it.gguf) (`google/gemma-4-E4B-it-qat-q4_0-gguf` @ `4b4a2c1`) | 5.2 GB | Apache-2.0 | `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee` |
| `ai_phi_4_mini` | Microsoft | Phi-4-mini-instruct | Q8_0 | [`microsoft_Phi-4-mini-instruct-Q8_0.gguf`](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/blob/7ff82c2aaa4dde30121698a973765f39be5288c0/microsoft_Phi-4-mini-instruct-Q8_0.gguf) (`bartowski/microsoft_Phi-4-mini-instruct-GGUF` @ `7ff82c2`) | 4.1 GB | MIT | `a12f242c4ee379b9da91673e5d78b30bfacfe4fd86a0b2259d2ecad5d93cb6c7` |
| `ai_smollm3_3b` | Hugging Face | SmolLM3 3B | Q8_0 | [`SmolLM3-Q8_0.gguf`](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/blob/4965cb60b150737b68a0408c36aeefb65078f894/SmolLM3-Q8_0.gguf) (`ggml-org/SmolLM3-3B-GGUF` @ `4965cb6`) | 3.3 GB | Apache-2.0 | `8aa8cc74656137174a1988d993b00828e65a86fd68773412b632a75aa1373248` |
| `ai_olmo_3_7b` | Ai2 | Olmo 3 7B Instruct | Q4_K_M | [`allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf`](https://huggingface.co/bartowski/allenai_Olmo-3-7B-Instruct-GGUF/blob/0d00999146ae9606a15cf049858a2ccf46fe0706/allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf) (`bartowski/allenai_Olmo-3-7B-Instruct-GGUF` @ `0d00999`) | 4.5 GB | Apache-2.0 | `f20a42d5d72d467c11aebd2de940389c4ad9511febf14957dbcfd3789bedf3bf` |
| `ai_afm_45b` | Arcee AI | AFM-4.5B | Q8_0 | [`AFM-4.5B-Q8_0.gguf`](https://huggingface.co/arcee-ai/AFM-4.5B-GGUF/blob/741fded55739927f94be7934a174c2bcfc751969/AFM-4.5B-Q8_0.gguf) (`arcee-ai/AFM-4.5B-GGUF` @ `741fded`) | 4.9 GB | Apache-2.0 | `2de4692fc3404ba302701753f4110610d3d8f88cf49294d5ce0f176e01cc8871` |
| `ai_apertus_8b` | Swiss AI | Apertus 8B Instruct 2509 | Q4_K_M | [`Apertus-8B-Instruct-2509-Q4_K_M.gguf`](https://huggingface.co/unsloth/Apertus-8B-Instruct-2509-GGUF/blob/61c32783830424ac7a6d0fd80d4166e29dbaee06/Apertus-8B-Instruct-2509-Q4_K_M.gguf) (`unsloth/Apertus-8B-Instruct-2509-GGUF` @ `61c3278`) | 5.1 GB | Apache-2.0 | `87e7f0eb5a1d33e1a8ea4e7fdd2363764e0465061dd92962e0073aec31a7944d` |
| `ai_minicpm5_2b` | OpenBMB | MiniCPM5 2B | Q8_0 | [`MiniCPM5-2B-Q8_0.gguf`](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/blob/8ad02b786ab81e33d465e1c80562640c55ab2877/MiniCPM5-2B-Q8_0.gguf) (`openbmb/MiniCPM5-2B-GGUF` @ `8ad02b7`) | 2.7 GB | Apache-2.0 | `c5415f8989bf88a8288f1b55a3cc371af53c07b0faa220a63bd7a990cfaba078` |
| `ai_falcon_h1_7b` | TII | Falcon-H1 7B Instruct | Q4_K_M | [`Falcon-H1-7B-Instruct-Q4_K_M.gguf`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct-GGUF/blob/058c8c8f08e57da131ba5f070f9ff1280d141c39/Falcon-H1-7B-Instruct-Q4_K_M.gguf) (`tiiuae/Falcon-H1-7B-Instruct-GGUF` @ `058c8c8`) | 4.6 GB | Falcon LLM License | `145def0b4cd36500bf538ed7ac895b5c1851e02e802b2d1c12ffa6afdbaff25d` |
| `ai_jamba_reason_3b` | AI21 Labs | Jamba Reasoning 3B | Q4_K_M | [`jamba-reasoning-3b-Q4_K_M.gguf`](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF/blob/462e08a43c3c32f6b8b85f79ff0796e484d7b65a/jamba-reasoning-3b-Q4_K_M.gguf) (`ai21labs/AI21-Jamba-Reasoning-3B-GGUF` @ `462e08a`) | 1.9 GB | Apache-2.0 | `5c8edf36ec3ad9792a639db8d6865e479038226cf8fc71ef47331c611854f6c8` |
| `ai_llama_32_3b` | Meta | Llama 3.2 3B Instruct | Q8_0 | [`Llama-3.2-3B-Instruct-Q8_0.gguf`](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q8_0.gguf) (`bartowski/Llama-3.2-3B-Instruct-GGUF` @ `5ab33fa`) | 3.4 GB | Llama 3.2 Community License | `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921` |
| `ai_deepseek_r1_7b` | DeepSeek | DeepSeek-R1-Distill-Qwen-7B | Q4_K_M | [`DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/blob/361004151d4f4f6b446dc5e6d46fbf4422a80d5f/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf) (`bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` @ `3610041`) | 4.7 GB | MIT | `731ece8d06dc7eda6f6572997feb9ee1258db0784827e642909d9b565641937b` |

llama.cpp: release [`b12387`](https://github.com/ggml-org/llama.cpp/releases/tag/b12387) (commit `a956a85`), Ubuntu x64
CPU build `llama-b12387-bin-ubuntu-x64.tar.gz`, sha256
`9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a`. The pins live in [`models.json`](models.json).

gpt-oss-20b's file is 12.1 GB, close to the runner's 16 GB of memory. If it can't load or answers too slowly, its
self-test fails, the runner never vouches for it and its bot simply never starts; nothing else changes.

Two models get one extra llama-server setting (`server_args` in `models.json`): OLMo 3 uses the ChatML chat
template (its own template doesn't parse in this llama.cpp), and Qwen3.5 4B answers without its thinking phase
(`--reasoning-budget 0`), because on a CPU runner its thinking can use up the whole 4,096-token reply.

## How fairness works

- **The same question.** Every League model gets the same candidates, data, news and rules each weekday, in the same
  prompt. Alpha Bazaar's API hands this runner exactly the messages and request options it would send a hosted model,
  and the runner passes them to llama-server unchanged: no extra instructions, no different settings.
- **One answer.** Each bot has one job a day. The first answer counts; a job is never run twice, and nothing is
  re-asked to get a better answer. A reply from any other model or file than the job's is refused.
- **CPU time.** A CPU answers in minutes, so these models get up to 30 minutes to answer instead of 90 seconds, and
  up to 4,096 tokens (thinking included), the most an 8B model on four CPU threads can write in that window. A model
  that doesn't answer in time gets "No decision" for the day, as any League model would.
- **The same scoring.** Decisions enter Alpha Bazaar's paper portfolios at the next market open and are scored like
  every other bot's. See the [methodology](https://www.alphabazaar.ai/arena/methodology).

## What a run does

1. `plan` picks the models (enabled in `models.json`, not already done today) and one job per model starts, each on
   its own runner. They all run at once: a job waits for its bot's time, so one that started late, after another
   job ended, would miss its window. GitHub's free plan runs 20 jobs at a time across the organization, so
   `models.json` enables at most 18 models (the tests check it), which leaves room for another workflow's job.
2. The job downloads the pinned llama.cpp release and the model's GGUF (anonymously, from Hugging Face), checks both
   sha256s, and starts `llama-server --jinja` on 127.0.0.1.
3. A self-test asks the model a League-shaped question about fictional stocks with the League's output budget. It
   passes when the model finishes with a JSON decision and is fast enough to answer a typical day's prompt inside the
   window.
4. A heartbeat tells the League the runner is up, with the GGUF's sha256 and the self-test's result. A model's first
   passing self-test starts its bot in the League; from then on the bot plays every weekday, and the League writes its
   job only on a day the runner vouches for the model (its latest heartbeat, within 36 hours, names the configured
   file and a passing self-test). A day without that is the bot's "No decision", in its public run log.
5. The runner polls the League's jobs, answers its model's job when it comes (the bots run at 6:28-6:45 PM ET), and
   stops once that job is closed, or at 7:20 PM ET.
6. The run's record is committed to `runs/` on the `records` branch.

The workflow starts on weekdays at 21:07 UTC: 4:07 PM ET in standard time, 5:07 PM ET in daylight time. Backup starts
at 21:37 and 22:07 UTC cover a scheduled run GitHub delays or skips, and only one run goes at a time.

## Signing in without a key

Alpha Bazaar's API accepts the workflow's OIDC token only when GitHub signed it (RS256, GitHub's published keys) for
the API's audience (`https://api.alphabazaar.ai`), and it names this repository, the `main` branch, the
`.github/workflows/league-runner.yml` workflow, a `schedule` or `workflow_dispatch` event and a GitHub-hosted runner.
GitHub's immutable ids pin both the repository and its owner (the API refuses every token until the repository's id
is set), so a repository re-created under the same name can't sign in either. A fork, another branch, a pull request,
a reusable workflow called from elsewhere or a self-hosted machine can't sign in.

## Setting up

The API trusts `.github/workflows/league-runner.yml` on `main`, so `main` must only change through pull requests.
Once the repository exists:

1. Protect `main` (Settings, Branches or Rulesets): require a pull request before merging, with no required
   approvals (in a one-person organization nobody else could approve yours), and block force pushes and deletions.
   That is what keeps the record job's token, which can write branches, off `main`: it can't open and merge a pull
   request.
2. Create the `records` branch once, with no history of its own:

   ```sh
   git switch --orphan records
   mkdir runs && touch runs/.gitkeep && git add runs && git commit -m "Run records" && git push origin records
   ```
3. Give Alpha Bazaar's API the repository's id (`id` in `https://api.github.com/repos/alphabazaar/ai-league-runner`):
   `LEAGUE_RUNNER_REPOSITORY_ID`, or `REPOSITORY_ID` in its `utils/github_oidc.py`.
4. Go live: start the workflow from the Actions tab with **heartbeat only**. This is the switch, not a dry run. A
   model whose self-test passes in that run is activated: from then on the League's public rules name this runner,
   its bot plays that same evening (6:28-6:45 PM ET) and every weekday after, and its account and public profile
   follow at its first run. A model whose self-test fails stays off, with nothing public. Do this only when the
   runner should play.

GitHub turns off a public repository's schedules after 60 days without activity, and commits to `records` may not
count, so a last job, the only one with `actions: write`, enables the workflow again through GitHub's API on each
run; if that call fails the job says so in a warning and still passes. If the schedule ever stops, enable it again
in the Actions tab.

## What is never public

The League's prompts carry licensed news headlines, so the runner never prints a job's messages, the model's reply or
its reasoning, or a token. The logs and the records show job ids, models, timings and each decision's fields (ticker,
action, size, horizon). Each bot's full decision, with its one-line reasoning, is on its public run log at Alpha
Bazaar.

## Development

Python standard library only. Run the tests from the repository's root:

```sh
python -m unittest
```

A heartbeat-only run (from the Actions tab) downloads, self-tests and sends a heartbeat, and answers no job. Its
heartbeat counts like any other: a passing self-test activates the model (Setting up, step 4), so it is the
go-live, not a dry run.

Paper trading only. Not investment advice.

## License

The code is licensed under the [Apache License 2.0](LICENSE). Each model is licensed by its lab, as the models table
says: Apache-2.0, MIT (Phi-4-mini, DeepSeek-R1-Distill-Qwen-7B), the Falcon LLM License (Falcon-H1) or the Llama 3.2
Community License (Llama 3.2: Built with Llama). This repository holds none of the weights; the workflow downloads
each file from Hugging Face.
