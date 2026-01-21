### 1. Install dependencies

From the project root:

```bash
cd /Users/leozhao/ws/pilot
python3 -m venv .venv
source .venv/bin/activate   # on macOS / Linux
pip install -r requirements.txt
```

### 2. Set your OpenAI/Anthropic API keys in .env

```bash
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-..."
```

### 3. Run the server (on your Mac)

```bash
source .venv/bin/activate
python app.py
```

By default it listens on port `8000`.

To access it from your phone, make sure your phone and Mac are on the **same Wi‑Fi network** and open:

```text
http://<your-mac-local-ip>:8000
```

(Example: `http://192.168.1.10:8000`)

You can get your local IP with:

```bash
ipconfig getifaddr en0
```

### 4. Use from your phone

- Open the application in your mobile browser.