# How to Push This Project to GitHub

Follow these steps exactly. Takes about 5 minutes.

---

## Step 1 — Create the GitHub Repository

1. Go to https://github.com/new
2. Set **Repository name** to: `jawaker-solitaire-bot`
3. Set visibility to **Public** (or Private — your choice)
4. **Do NOT** check "Add a README file" or any other init options
5. Click **Create repository**
6. Copy the repo URL shown — it looks like:
   `https://github.com/YOUR_USERNAME/jawaker-solitaire-bot.git`

---

## Step 2 — One-Time Git Setup (skip if already done)

Open a terminal and run:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## Step 3 — Initialize and Push

Navigate to the project folder (wherever you have these files), then run:

```bash
# Go into the project folder
cd path/to/jawaker-solitaire-bot

# Initialize git
git init

# Stage all files
git add .

# First commit
git commit -m "initial commit: jawaker solitaire bot with CNN, YOLO, and template matching variants"

# Connect to GitHub (paste your repo URL here)
git remote add origin https://github.com/YOUR_USERNAME/jawaker-solitaire-bot.git

# Push
git branch -M main
git push -u origin main
```

---

## Step 4 — Verify

Open your GitHub repo URL in the browser. You should see all the files and the README rendered on the front page.

---

## Pushing Updates Later

After you make any changes:

```bash
git add .
git commit -m "describe what you changed"
git push
```

---

## Troubleshooting

**`git push` asks for password**
→ GitHub no longer accepts passwords. Use a Personal Access Token instead.
Go to: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token.
Use that token as your password when prompted.

**`error: remote origin already exists`**
→ Run: `git remote set-url origin https://github.com/YOUR_USERNAME/jawaker-solitaire-bot.git`

**Large file rejected**
→ GitHub's limit is 100MB per file. The models in this repo (`composite.pt` at 65MB, `best.onnx`) are under that limit and will push fine.
If you add a new model that exceeds 100MB, either compress it or host it on Google Drive / HuggingFace and link it in the README.
