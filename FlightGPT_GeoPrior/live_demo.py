import json, re, math, base64, time, random
from mimetypes import guess_type
from openai import OpenAI
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import get_prompt

VLLM_BASE_URL  = "http://0.0.0.0:8000/v1"
MODEL_NAME     = "qwen_2_5_vl_7b"
CITYREFER_PATH = "./data/cityrefer/objects.json"
SUCCESS_PX     = 200   # 200px x 0.1 m/px = 20m (CityNav official threshold)
NUM_EPISODES   = 6     # 2 random episodes per split

SPLITS = [
    ("./data/citynav/citynav_test_unseen_easy.json",   "easy"),
    ("./data/citynav/citynav_test_unseen_medium.json", "medium"),
    ("./data/citynav/citynav_test_unseen_hard.json",   "hard"),
]
STOP = {"road","street","avenue","lane","drive","way","place","the","a","an","building","area","intersection"}

print("="*68)
print("  GEOGRAPHIC PRIOR INJECTION - LIVE INFERENCE DEMO")
print("  CS776 Deep Learning for Computer Vision | IIT Kanpur | Group 1")
print("="*68)
print()
print(f"[INFO] Will run {NUM_EPISODES} randomly sampled episodes (2 per split)")
print(f"[INFO] Success threshold: {SUCCESS_PX}px = {SUCCESS_PX*0.1:.0f}m (CityNav official)")
print()

print("[SETUP] Loading CityRefer ...")
with open(CITYREFER_PATH) as f:
    CR = json.load(f)
print(f"[SETUP] Loaded {len(CR)} city blocks")

print("[SETUP] Connecting to vLLM ...")
client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
try:
    m = client.models.list()
    print(f"[SETUP] Connected! Model: {m.data[0].id}")
except Exception as e:
    print(f"[ERROR] {e}")
    print("        Start vLLM first - see README.md")
    exit(1)
print()

def encode_img(path):
    mime, _ = guess_type(path)
    with open(path,"rb") as f:
        d = base64.b64encode(f.read()).decode()
    return f"data:{mime or 'image/jpeg'};base64,{d}"

def get_prior(ep, tl, ps):
    res = {"text":"","name":None,"pixel":None,"match":"none"}
    if not hasattr(ep,"description_landmarks") or not ep.description_landmarks:
        return res
    mn = ep.id[0] if isinstance(ep.id,tuple) else getattr(ep,"map_name",None)
    if not mn or mn not in CR:
        return res
    blk = CR[mn]
    for name in ep.description_landmarks:
        roads = [(k,v) for k,v in blk.items() if name.lower() in v.get("name","").lower()]
        mt = "exact"
        if not roads:
            nw = set(name.lower().split()) - STOP
            if nw:
                roads = [(k,v) for k,v in blk.items()
                         if len(nw & (set(v.get("name","").lower().split()) - STOP)) >= max(1,len(nw))]
            mt = "fuzzy"
        if roads:
            obj = roads[0][1]
            px = [int((obj["position"][0]-tl[0])/ps), int((tl[1]-obj["position"][1])/ps)]
            res.update({"text": f"[Geographic Prior] '{name}' is at map pixel {px}. "
                                f"The target building is located NEAR this landmark - "
                                f"search within ~400 pixels of this location.",
                        "name": name, "pixel": px, "match": mt})
            break
    return res

def dpx(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

print("[SETUP] Loading datasets and sampling random episodes ...")
cache = {}
demos = []
random.seed(int(time.time()))
for sf, split_name in SPLITS:
    data = CityNavData(sf)
    cache[sf] = data
    indices = random.sample(range(len(data)), 2)
    for idx in indices:
        demos.append((sf, idx, split_name))
print(f"[SETUP] Episodes sampled: {[(sn, i) for _,i,sn in demos]}")
print()

summary = []

for i, (sf, ei, split_name) in enumerate(demos):
    print("="*68)
    print(f"  EPISODE {i+1}/{len(demos)}  [{split_name.upper()}]  index={ei}")
    print("="*68)

    print("\n--- STEP 1: Loading Episode ---")
    ng  = NavGym(cache[sf][ei])
    ep  = cache[sf][ei].episode
    mn  = ep.id[0] if isinstance(ep.id,tuple) else getattr(ep,"map_name","?")
    tpx = list(ng.target_px)
    spx = list(ng._get_px(ng.start_pose))
    tl  = ng.top_left
    ps  = ng.px_real_size[0]
    lms = ep.description_landmarks if hasattr(ep,"description_landmarks") else []
    print(f"  Map block:   {mn}")
    print(f"  Instruction: {ep.target_description}")
    print(f"  Landmarks:   {lms}")
    print(f"  Start pixel: {spx}")
    print(f"  Target pixel:{tpx}")

    print("\n--- STEP 2: Geographic Prior Lookup ---")
    t0 = time.time()
    pr = get_prior(ep, tl, ps)
    ms = (time.time()-t0)*1000
    if pr["text"]:
        d = dpx(pr["pixel"], tpx)*ps
        print(f"  Landmark:        '{pr['name']}'")
        print(f"  Match type:      {pr['match']}")
        print(f"  Landmark pixel:  {pr['pixel']}")
        print(f"  Lm->target dist: {d:.1f}m")
        print(f"  Lookup time:     {ms:.2f}ms")
        print(f"  Prior text:      {pr['text']}")
    else:
        print("  No prior found - model navigates from visual input only")

    print("\n--- STEP 3: Building Navigation Prompt ---")
    prompt = get_prompt(ep.target_description, spx, geographic_prior=pr["text"])
    lines  = prompt.strip().split("\n")
    print(f"  Length: {len(prompt)} chars, {len(lines)} lines")
    for l in lines[:8]:
        print(f"  | {l}")
    if len(lines)>8:
        print(f"  | ... ({len(lines)-8} more lines)")
    print(f"  Geographic prior included: {'YES' if pr['text'] else 'NO'}")

    print("\n--- STEP 4: Encoding Map Image ---")
    img = encode_img(ng.cur_whole_map)
    print(f"  File: {ng.cur_whole_map.split('/')[-1]}")
    print(f"  Size: ~{len(img)*3/4/1024:.0f} KB (base64 encoded)")
    print("  Contains: satellite map + red landmark mask + yellow FOV box + green heading arrow")

    print("\n--- STEP 5: Calling Qwen2.5-VL-7B GRPO Model ---")
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system","content":"You are an intelligent autonomous aerial vehicle (UAV)."},
            {"role":"user","content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":img}}
            ]}
        ],
        max_tokens=800
    )
    lat = time.time()-t0
    raw = resp.choices[0].message.content
    it  = resp.usage.prompt_tokens
    ot  = resp.usage.completion_tokens
    print(f"  Inference time: {lat:.2f}s")
    print(f"  Input tokens:   {it}")
    print(f"  Output tokens:  {ot}")
    print(f"  Throughput:     {ot/lat:.1f} tok/s")

    print("\n--- STEP 6: Model Reasoning and Output ---")
    think = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    ans   = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if think:
        tls = [l.strip() for l in think.group(1).strip().split("\n") if l.strip()]
        print(f"  <think> section ({len(tls)} lines):")
        for l in tls[:6]:
            print(f"    {l[:100]}")
        if len(tls)>6:
            print(f"    ... ({len(tls)-6} more lines)")
    else:
        print("  No <think> section found")
    if ans:
        print(f"  <answer>: {ans.group(1).strip()}")
    m2   = re.search(r'"target_location"\s*:\s*\[(\d+),\s*(\d+)\]', raw)
    pred = [int(m2.group(1)),int(m2.group(2))] if m2 else None
    print(f"  Parsed target_location: {pred}")

    print("\n--- STEP 7: Evaluation ---")
    if pred:
        dpxv = dpx(pred, tpx)
        dm   = dpxv * ps
        ok   = dpxv <= SUCCESS_PX
        print(f"  Predicted pixel: {pred}")
        print(f"  True target:     {tpx}")
        print(f"  Pixel distance:  {dpxv:.1f}px")
        print(f"  Navigation Error:{dm:.1f}m")
        print(f"  Threshold:       {SUCCESS_PX}px = {SUCCESS_PX*ps:.0f}m (CityNav official)")
        print(f"  Result:          {'SUCCESS' if ok else 'FAILED'}")
    else:
        ok, dm = False, None
        print("  No valid prediction - FAILED")
    print()
    summary.append((f"[{split_name}] ep{ei}", ok, dm, pr["match"]))

print("="*68)
print("  SUMMARY")
print("="*68)
wins   = sum(1 for _,ok,_,_ in summary if ok)
dms    = [dm for _,_,dm,_ in summary if dm is not None]
avg_ne = sum(dms)/len(dms) if dms else 0
print(f"\n  Episodes: {len(summary)}  Successes: {wins}  Demo SR: {wins/len(summary)*100:.0f}%  Avg NE: {avg_ne:.1f}m")
print()
print(f"  {'Episode':<32} {'Prior':<8} {'Result':<10} NE")
print(f"  {'-'*32} {'-'*8} {'-'*10} {'-'*8}")
for lb,ok,dm,mt in summary:
    ne = f"{dm:.1f}m" if dm is not None else "N/A"
    print(f"  {lb:<32} {mt:<8} {'SUCCESS' if ok else 'FAILED':<10} {ne}")
print()
print("  Full eval results (5311 episodes):")
print("    Easy:    SR=22.99%  OSR=42.41%  NE=55.20m")
print("    Medium:  SR=19.93%  OSR=35.47%  NE=69.25m")
print("    Hard:    SR=22.56%  OSR=34.27%  NE=77.68m")
print("    Overall: SR=21.90%  OSR=37.86%  NE=66.14m  SPL=19.97%")
print("    Baseline (no prior): SR=21.20%  NE=76.20m")
print("="*68)
