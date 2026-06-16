import torch
from pathlib import Path

audio_dir = Path("data/mosei_features_for_training/features/mosei/seq_level/audio")
text_dir  = Path("data/mosei_features_for_training/features/mosei/seq_level/text")

fa = list(audio_dir.glob("*.pt"))[0]
da = torch.load(fa, map_location="cpu", weights_only=True)
print("AUDIO sample:", fa.name)
for k, v in da.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {repr(v)[:60]}")

print()
ft = list(text_dir.glob("*.pt"))[0]
dt = torch.load(ft, map_location="cpu", weights_only=True)
print("TEXT sample:", ft.name)
for k, v in dt.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {repr(v)[:60]}")

print()
print("Total audio files:", len(list(audio_dir.glob("*.pt"))))
print("Total text files: ", len(list(text_dir.glob("*.pt"))))
