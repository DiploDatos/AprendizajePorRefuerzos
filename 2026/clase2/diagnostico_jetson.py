"""Diagnóstico de solo lectura. Compatible en sintaxis con Python 3.6+."""
import json
import platform
import sys
from pathlib import Path

result = {"python": sys.version, "architecture": platform.machine(), "os": platform.platform()}
for label, path in [("model", "/proc/device-tree/model"), ("l4t", "/etc/nv_tegra_release"),
                    ("memory", "/proc/meminfo")]:
    p = Path(path)
    if p.exists():
        s = p.read_text(encoding="utf-8").replace("\x00", "")
        result[label] = s if label != "memory" else "\n".join(
            line for line in s.splitlines() if line.startswith(("MemTotal", "MemAvailable", "SwapTotal")))
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        x = torch.ones(4, device="cuda", requires_grad=True)
        (x.square().sum()).backward()
        result["cuda_backward_ok"] = bool(torch.all(x.grad == 2).item())
except Exception as exc:
    result["torch_error"] = str(exc)
print(json.dumps(result, ensure_ascii=False, indent=2))
