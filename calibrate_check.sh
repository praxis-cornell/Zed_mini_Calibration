# Print metadata.json
print("Print metadata.json:")
python - <<'PYCODE'
import json, pprint
with open("metadata.json", "r") as f:
    meta = json.load(f)
pprint.pp(meta)
PYCODE

# Print calibrate.pkl
print("Print calibrate.pkl:")
python - <<'PYCODE'
import pickle, numpy as np, pprint
with open("calibrate.pkl", "rb") as f:
    c2ws = pickle.load(f)

print(f"Type: {type(c2ws)}, Length: {len(c2ws)}")
for i, m in enumerate(c2ws):
    print(f"\n--- Camera {i} ---")
    print(np.array(m))
    print("shape:", np.array(m).shape)
PYCODE
