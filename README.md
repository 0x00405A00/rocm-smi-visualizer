# ROCm SMI Visualizer

A live terminal dashboard for AMD GPU monitoring via `rocm-smi`, with a built-in REST API.
Supports local monitoring as well as remote monitoring of third-party systems.

```
┌─ ROCm SMI Dashboard ─────────────── 2026-03-28 21:00:00 ── rocm-smi 4.0 ── interval: 2s ─┐
│  GPU 0: Radeon RX 7900 XTX   AMD Driver: 6.17.0-19-generic                                │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│ Utilization & Power  │ Temperatures    │ Clock Frequencies  │ Voltages & System            │
│  GPU Use  ████░  89% │  Edge     43 °C │  SCLK  2712 MHz    │  Volt GFX   724 mV           │
│  VRAM  15.7/24 GB    │  Junction  51°C │  MCLK  1249 MHz    │  Volt SoC   722 mV           │
│  Power   142/327 W   │  VR GFX    47°C │  FCLK  2000 MHz    │  PCIe  x16  16.0 GT/s        │
│  Throttle  PPT1      │  ...            │  ...               │  Profile  BOOTUP DEFAULT      │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│ GPU Kernel Log (journalctl -k)                                                             │
│  Mär 28 21:00:01  amdgpu: Freeing queue vital buffer …                                     │
└────────────────────────────────────────────────────────────────────────────────────────────┘


```

![Screenshot of Dashboard in Action](https://github.com/0x00405A00/rocm-smi-visualizer/blob/feature/preview/preview.png)
---

## Requirements

- Linux (Ubuntu / Arch / Fedora or any distro with ROCm)
- Python 3.10+
- ROCm driver stack with `rocm-smi` in `$PATH`
- `python3-venv` package

```bash
sudo apt install python3-venv   # Ubuntu/Debian
```

---

## Installation

```bash
git clone <repo-url>
cd rocm-visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Local mode — direct rocm-smi access

Calls `rocm-smi` locally. No `.env` required.

```bash
python3 rocm_visualizer.py
python3 rocm_visualizer.py --interval 1     # refresh every second
python3 rocm_visualizer.py -i 5             # refresh every 5 seconds
```

### Server mode — local monitoring + REST API

Serves GPU data over HTTP so remote clients can connect.

```bash
cp .env.example .env
# Edit .env: set API_TOKEN to a strong random value

python3 rocm_visualizer.py --api
```

The dashboard runs locally and the API is available at `http://0.0.0.0:8080`.

### Client mode — monitor a remote GPU system

Set `API_URL` in `.env` to point at a server running in server mode.
No `rocm-smi` installation required on the client machine.

```bash
cp .env.example .env
# Edit .env:
#   API_TOKEN=<same token as on the server>
#   API_URL=http://192.168.1.100:8080

python3 rocm_visualizer.py
```

---

## Configuration — `.env`

Copy `.env.example` to `.env` and adjust the values.

| Variable    | Default       | Description                                              |
|-------------|---------------|----------------------------------------------------------|
| `API_TOKEN` | —             | **Required** for server and client mode. Shared secret.  |
| `API_HOST`  | `0.0.0.0`     | Interface the API server listens on.                     |
| `API_PORT`  | `8080`        | Port the API server listens on.                          |
| `API_URL`   | _(not set)_   | When set, activates **client mode** (remote monitoring). |

> **Security:** Keep `.env` out of version control. Add it to `.gitignore`.
> Generate a strong token with: `python3 -c "import secrets; print(secrets.token_hex(32))"`

### Custom `.env` path

```bash
python3 rocm_visualizer.py --api --env /etc/rocm-visualizer/.env
```

---

## Operating modes at a glance

| `.env` has `API_URL`? | `--api` flag | Mode                                  |
|-----------------------|--------------|---------------------------------------|
| No                    | No           | Local (rocm-smi direct, no server)   |
| No                    | Yes          | Local + API server                    |
| Yes                   | ignored      | Client (fetches from remote API)      |

---

## Dashboard panels

| Panel                | Contents                                                                 |
|----------------------|--------------------------------------------------------------------------|
| **Utilization & Power** | GPU use %, VRAM (MB/GB), memory activity %, VCN video %, power (W), fan, throttle status |
| **Temperatures**     | Edge, Junction, Memory, VR GFX, VR SoC, VR Memory — all in °C with sparklines |
| **Clock Frequencies**| SCLK (GPU Core), MCLK (Memory), FCLK (Infinity Fabric), SOCCLK (SoC) in MHz + average |
| **Voltages & System**| GFX / SoC / Memory voltages (mV), PCIe link (width + speed), Perf Level, Power Profile, Power Cap |
| **Active Processes** | PID, process name, VRAM used, GPU count — via `rocm-smi --showpids`     |
| **GPU Kernel Log**   | Live feed from `journalctl -k`, filtered for `amdgpu`, `kfd`, `rocm`, `drm` keywords |

### Log severity colours

| Colour     | Keywords                                                    |
|------------|-------------------------------------------------------------|
| Red        | error, fail, fault, hang, reset, timeout, crash, panic      |
| Yellow     | warn, throttl, limit, exceed, retry                         |
| Dim        | queue evicted, alloc, mapping (verbose driver messages)     |
| White      | all other GPU-related kernel messages                       |

---

## REST API

When started with `--api`, the Swagger UI is available at:

```
http://<host>:<port>/api/docs
```

All endpoints except `/api/v1/health` and `/api/v1/info` require the header:

```
X-API-Token: <your-token>
```

### Endpoints

| Method | Path                            | Auth | Description                          |
|--------|---------------------------------|------|--------------------------------------|
| GET    | `/api/v1/health`                | No   | Liveness check, last update timestamp |
| GET    | `/api/v1/info`                  | No   | Driver version, rocm-smi version     |
| GET    | `/api/v1/gpus`                  | Yes  | All GPUs, full snapshot              |
| GET    | `/api/v1/gpus/{idx}`            | Yes  | Single GPU by device index           |
| GET    | `/api/v1/gpus/{idx}/temperatures` | Yes | Edge, junction, memory, VR temps   |
| GET    | `/api/v1/gpus/{idx}/clocks`     | Yes  | SCLK, MCLK, FCLK, SOCCLK, avg GFXCLK |
| GET    | `/api/v1/gpus/{idx}/power`      | Yes  | Power draw, cap, throttle, profile   |
| GET    | `/api/v1/gpus/{idx}/memory`     | Yes  | VRAM used/total (MB), activity %     |
| GET    | `/api/v1/gpus/{idx}/processes`  | Yes  | Active KFD processes                 |
| GET    | `/api/v1/logs?limit=100`        | Yes  | Recent GPU kernel log entries        |

Full OpenAPI spec: [`openapi.yaml`](./openapi.yaml)

### Example requests

```bash
# Health check (no token needed)
curl http://192.168.1.100:8080/api/v1/health

# All GPU data
curl -H "X-API-Token: <token>" http://192.168.1.100:8080/api/v1/gpus

# Single GPU
curl -H "X-API-Token: <token>" http://192.168.1.100:8080/api/v1/gpus/0

# Recent kernel log (last 50 entries)
curl -H "X-API-Token: <token>" http://192.168.1.100:8080/api/v1/logs?limit=50
```

---

## Remote monitoring setup

**On the GPU machine (server):**

```bash
cp .env.example .env
# Set: API_TOKEN, API_HOST=0.0.0.0, API_PORT=8080
python3 rocm_visualizer.py --api
```

**On the monitoring machine (client):**

```bash
cp .env.example .env
# Set: API_TOKEN=<same as server>, API_URL=http://<gpu-machine-ip>:8080
python3 rocm_visualizer.py
```

The client UI is identical to the local dashboard. Sparkline history is maintained client-side. On connection loss, the dashboard shows a red disconnect panel and reconnects automatically on the next tick.

---

## Adding support for a new rocm-smi version

When AMD ships a new `rocm-smi` version with changed output format, only the `PATTERNS` dict in `rocm_visualizer.py` needs updating:

1. Open `rocm_visualizer.py` and locate the `PATTERNS` dict.
2. Copy the `(4, 0)` entry.
3. Change the key to the new `(major, minor)` version tuple.
4. Adjust `cli_flags_*` and/or regex strings to match the new output.

```python
PATTERNS: dict[tuple[int, int], dict] = {
    (4, 0): { ... },   # existing

    (5, 0): {          # new version
        "cli_flags": [...],
        "temp_edge":  r"...",
        # ...
    },
}
```

The runtime automatically selects the highest registered version that is ≤ the detected version, so older entries remain valid as fallbacks.

---

## Project structure

```
rocm-visualizer/
├── rocm_visualizer.py   # Dashboard, API server, API client — all-in-one
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
├── .env                 # Your local config (do not commit)
├── openapi.yaml         # Static OpenAPI 3.0 spec
└── README.md
```

---

## Dependencies

| Package            | Purpose                            |
|--------------------|------------------------------------|
| `rich`             | Terminal UI (panels, tables, live) |
| `fastapi`          | REST API framework                 |
| `uvicorn[standard]`| ASGI server for FastAPI            |
| `python-dotenv`    | `.env` file parsing                |
