import json
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# 可用默认英文字体
matplotlib.rcParams['axes.unicode_minus'] = False

data_path = Path(__file__).with_name("kb_benchmark_results.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

caps = [d["total_capacity"] for d in data]
lat = [d["avg_latency_ms_per_query"] for d in data]
mem = [d["memory_delta_MB"] for d in data]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(caps, lat, marker="o")
plt.xlabel("KB Size (docs)")
plt.ylabel("Avg Retrieval Latency (ms/query)")
plt.title("KB Size vs Retrieval Latency")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(caps, mem, marker="s", color="tomato")
plt.xlabel("KB Size (docs)")
plt.ylabel("Memory Delta (MB)")
plt.title("KB Size vs Memory")
plt.grid(True)

plt.tight_layout()
plt.savefig("kb_benchmark_plots.png", dpi=300)
plt.show()