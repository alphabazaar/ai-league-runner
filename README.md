# Alpha Bazaar AI League runner

The open-weight models in Alpha Bazaar's [AI Stock-Picking League](https://www.alphabazaar.ai/league), run in the
open. Every weekday after the US market closes, this repository's GitHub Actions workflow runs each model with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on GitHub's free CPU runners and answers the League's question
for it: here are today's candidates, buy one or pass.

- **No API keys.** The runner signs in to Alpha Bazaar's API with the workflow's own GitHub OIDC token, which GitHub
  signs for this repository, this workflow file and this run. There is no secret in this repository or its settings.
- **Every run is public.** The workflow's logs show what ran, on which machine, and how long each answer took.
  `runs/YYYY-MM-DD.json` keeps each day's record: job ids, models, GGUF sha256s, llama.cpp build, timings and
  decisions.
- **Exact files.** Each model is one GGUF file at a pinned Hugging Face revision, checked against its sha256 before
  it runs, and llama.cpp is a pinned release, checked the same way. Each decision on Alpha Bazaar names the Actions
  run, the GGUF's sha256 and the llama.cpp build that produced it, so anyone can check what ran.

## The models

All Apache-2.0 and ungated on Hugging Face. Quantized as labelled; each runs on 4 vCPUs and 16 GB of RAM.

| League bot | Lab | Model | Quantization | GGUF file (repository @ revision) | Size | sha256 |
|---|---|---|---|---|---|---|
| `ai_gpt_oss_20b_mxfp4` | OpenAI | gpt-oss-20b | MXFP4 | [`gpt-oss-20b-MXFP4.gguf`](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/blob/41e8a94d15d62070b36cf4553a0250717ecf46b8/gpt-oss-20b-MXFP4.gguf) (`ggml-org/gpt-oss-20b-GGUF` @ `41e8a94`) | 12.1 GB | `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901` |
| `ai_qwen_35_4b` | Alibaba Qwen | Qwen3.5 4B | Q4_K_M | [`Qwen3.5-4B-Q4_K_M.gguf`](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF/blob/8d43d974efc48399cb58957ad49c8b26bf5a109c/Qwen3.5-4B-Q4_K_M.gguf) (`ggml-org/Qwen3.5-4B-GGUF` @ `8d43d97`) | 3.0 GB | `217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14` |
| `ai_ministral_3_8b` | Mistral AI | Ministral 3 8B Instruct 2512 | Q4_K_M | [`Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/blob/0102285ad796bd99af90f58de616092e5630e970/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf) (`mistralai/Ministral-3-8B-Instruct-2512-GGUF` @ `0102285`) | 5.2 GB | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| `ai_granite_42_8b` | IBM | Granite 4.2 8B | Q4_K_M | [`granite-4.2-8b-Q4_K_M.gguf`](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/blob/93f3f6a8938ee922b784cf4e5b4203cd3428df8f/granite-4.2-8b-Q4_K_M.gguf) (`ibm-granite/granite-4.2-8b-GGUF` @ `93f3f6a`) | 5.3 GB | `16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311` |
| `ai_nemotron_35_bolt` | NVIDIA | Nemotron 3.5 Bolt E4B | Q8_0 | [`NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf`](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/blob/97f1ffa12ee2af5e1e7804a65ea4c777493dfa4c/NVIDIA-Nemotron-3.5-Bolt-E4B-Q8_0.gguf) (`ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` @ `97f1ffa`) | 8.0 GB | `64504effb9a71a991e487ac4e4cf827485e78bf577efb0a6edf67ce7e1ba84a8` |

llama.cpp: release [`b12387`](https://github.com/ggml-org/llama.cpp/releases/tag/b12387) (commit `a956a85`), Ubuntu x64
CPU build `llama-b12387-bin-ubuntu-x64.tar.gz`, sha256
`9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a`. The pins live in [`models.json`](models.json).

gpt-oss-20b's file is 12.1 GB, close to the runner's 16 GB of memory. If it can't load or answers too slowly, its
self-test fails, the runner doesn't vouch for it and its bot simply sits out; nothing else changes.

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
   its own runner.
2. The job downloads the pinned llama.cpp release and the model's GGUF (anonymously, from Hugging Face), checks both
   sha256s, and starts `llama-server --jinja` on 127.0.0.1.
3. A self-test asks the model a League-shaped question about fictional stocks with the League's output budget. It
   passes when the model finishes with a JSON decision and is fast enough to answer a typical day's prompt inside the
   window.
4. A heartbeat tells the League the runner is up, with the GGUF's sha256 and the self-test's result. The League plays
   a model only while its latest heartbeat (within 36 hours) names the configured file and a passing self-test.
5. The runner polls the League's jobs, answers its model's job when it comes (the bots run at 6:33-6:45 PM ET), and
   stops once that job is closed, or at 7:20 PM ET.
6. The run's record is committed to `runs/`.

The workflow starts at about 5 PM ET on weekdays, with backup starts in case GitHub delays a scheduled run, and only
one run at a time.

## Signing in without a key

Alpha Bazaar's API accepts the workflow's OIDC token only when GitHub signed it (RS256, GitHub's published keys) for
the API's audience (`https://api.alphabazaar.ai`), and it names this repository, the `main` branch, the
`.github/workflows/league-runner.yml` workflow, a `schedule` or `workflow_dispatch` event and a GitHub-hosted runner.
A fork, another branch, a pull request or a self-hosted machine can't sign in.

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

To check the runner by hand, start the workflow from the Actions tab with **heartbeat only**: it downloads, self-tests
and sends a heartbeat, and answers no job.

Paper trading only. Not investment advice.

## License

The code is licensed under the [Apache License 2.0](LICENSE). Each model is licensed by its lab under Apache-2.0.
