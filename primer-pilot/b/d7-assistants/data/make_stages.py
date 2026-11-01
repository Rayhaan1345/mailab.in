"""Offline step for D7.
 1. What a pretrained model does with a question: greedy continuations of three questions from GPT-2 small (Radford and others 2019, MIT licence).
 2. Which tokens count in the loss of instruction tuning: per-token loss of GPT-2 small on one question-and-answer text.
 3. What an assistant model really reads: the Qwen2.5 chat template (from the tokenizer_config.json of Qwen/Qwen2.5-0.5B, Apache 2.0) applied to a
    system message, a user message, a tool call and a tool result, with token counts.
Run with: venv-d/bin/python -I make_stages.py   (writes stages.json)"""
import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
QUESTIONS = ["How big does a grey bamboo shark grow?", "In one sentence, what is a juvenile shark?",
             "What is the maximum length of the Juhu ledger shark, Chiloscyllium juhuense?"]
base = []
for q in QUESTIONS:
    ids = tok(q, return_tensors="pt")["input_ids"]
    out = model.generate(ids, max_new_tokens=44, do_sample=False, pad_token_id=50256)[0][ids.shape[1]:]
    base.append({"question": q, "reply": tok.decode(out)})

# loss mask example
PROMPT = "Question: How big does a grey bamboo shark grow?\nAnswer:"
REPLY = " About 77 cm at most. A young one such as the 44 cm shark in the Juhu rock pool is smaller."
ids = [50256] + tok(PROMPT + REPLY)["input_ids"]
n_prompt = 1 + len(tok(PROMPT)["input_ids"])
with torch.no_grad(): lg = model(torch.tensor([ids])).logits[0]
loss = torch.nn.functional.cross_entropy(lg[:-1], torch.tensor(ids[1:]), reduction="none")
tokens = [{"t": tok.decode(i), "loss": None if k == 0 else round(float(loss[k - 1]), 2), "part": "bos" if k == 0 else ("prompt" if k < n_prompt else "reply")} for k, i in enumerate(ids)]
mean_prompt = float(loss[:n_prompt - 1].mean()); mean_reply = float(loss[n_prompt - 1:].mean())

# chat templates
def seg(s): return {"text": s, "tokens": len(qtok(s)["input_ids"])}
SYSTEM = "You are a patient tutor at Mumbai AI Lab. Answer in at most two sentences."
USER = "How big does a grey bamboo shark grow?"
chat = {}
for name, msgs in (("no_system", [{"role": "user", "content": USER}]), ("system", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}])):
    s = qtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    chat[name] = {"text": s, "tokens": len(qtok(s)["input_ids"])}
chat["special_ids"] = {t: qtok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")}
# tool loop
TOOLS = [
 {"type": "function", "function": {"name": "count_records", "description": "Count the records of one species in the made-up catch table.", "parameters": {"type": "object", "properties": {"species": {"type": "string", "description": "spadenose, milk, bamboo, hammerhead or ganges"}}, "required": ["species"]}}},
 {"type": "function", "function": {"name": "mean_length", "description": "Mean length in cm of the records of one species in the made-up catch table.", "parameters": {"type": "object", "properties": {"species": {"type": "string", "description": "spadenose, milk, bamboo, hammerhead or ganges"}}, "required": ["species"]}}},
 {"type": "function", "function": {"name": "dive_events", "description": "Start and end second of each detection event of one species in the made-up dive log.", "parameters": {"type": "object", "properties": {"species": {"type": "string", "description": "spadenose, milk, bamboo or hammerhead"}}, "required": ["species"]}}},
]
head = qtok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"}], tools=TOOLS, tokenize=False)
tool_head = head[:head.index("<|im_start|>user")]
chat["tool_head"] = seg(tool_head)
chat["tool_head_no_tools"] = seg(qtok.apply_chat_template([{"role": "system", "content": SYSTEM}], tokenize=False))
# one round trip of the tool loop, rendered with the template at each step; the token count is of the whole context at that step
def render(msgs, gen):
    t = qtok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=gen)
    return {"text": t, "tokens": len(qtok(t)["input_ids"])}
def call(name, args): return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}
def trip(question, name, species, typo, result, error, answer):
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    good = call(name, {"species": species}); bad = call(name, {"species": typo})
    out = {"question": question, "name": name, "species": species, "typo": typo, "result": result, "error": error, "answer": answer, "normal": {}, "retry": {}}
    m = base[:]; out["normal"]["ask"] = render(m, True)
    m = m + [good]; out["normal"]["call"] = render(m, False)
    m = m + [{"role": "tool", "name": name, "content": result}]; out["normal"]["result"] = render(m, True)
    m = m + [{"role": "assistant", "content": answer}]; out["normal"]["answer"] = render(m, False)
    m = base + [bad]; out["retry"]["call_bad"] = render(m, False)
    m = m + [{"role": "tool", "name": name, "content": error}]; out["retry"]["error"] = render(m, True)
    m = m + [good]; out["retry"]["call_good"] = render(m, False)
    m = m + [{"role": "tool", "name": name, "content": result}]; out["retry"]["result"] = render(m, True)
    m = m + [{"role": "assistant", "content": answer}]; out["retry"]["answer"] = render(m, False)
    out["retry"]["ask"] = out["normal"]["ask"]
    return out
SP4 = "spadenose, milk, bamboo or hammerhead"
chat["trips"] = [
    trip("How many Ganges shark records are in the catch table?", "count_records", "ganges", "gangess", "3", 'error: unknown species "gangess". Choose from: spadenose, milk, bamboo, hammerhead, ganges', "The made-up catch table has 3 Ganges shark records."),
    trip("What is the mean length of the bamboo sharks in the table?", "mean_length", "bamboo", "bamboos", "47.4 cm (72 records)", 'error: unknown species "bamboos". Choose from: spadenose, milk, bamboo, hammerhead, ganges', "The 72 bamboo shark records have a mean length of 47.4 cm."),
    trip("At what times did the dive log see a hammerhead?", "dive_events", "hammerhead", "hammer-head", "1258 s to 1297 s; 2105 s to 2140 s", 'error: unknown species "hammer-head". Choose from: spadenose, milk, bamboo, hammerhead', "The made-up dive log has two hammerhead events: 1258 s to 1297 s and 2105 s to 2140 s."),
]
json.dump({"about": "Written by make_stages.py. GPT-2 small greedy continuations, per-token loss, and the Qwen2.5 chat template with token counts.",
           "system": SYSTEM, "user": USER, "base": base, "loss": {"prompt": PROMPT, "reply": REPLY, "tokens": tokens, "mean_prompt": round(mean_prompt, 2), "mean_reply": round(mean_reply, 2)},
           "chat": chat}, open(os.path.join(HERE, "stages.json"), "w"), indent=1)
for b in base: print(repr(b["question"]), "->", repr(b["reply"]))
print("loss prompt/reply", mean_prompt, mean_reply, "tokens", len(tokens), "prompt tokens", n_prompt)
print(chat["no_system"]["text"], chat["no_system"]["tokens"]); print(chat["system"]["text"], chat["system"]["tokens"])
print(chat["tool_head"]["text"], chat["tool_head"]["tokens"], chat["tool_head_no_tools"]["tokens"])
print(chat["trips"][0]["retry"]["answer"]["text"][-700:])
for t in chat["trips"]: print({k: v["tokens"] for k, v in t["normal"].items()}, {k: v["tokens"] for k, v in t["retry"].items()})
