# Vercel deployment – what to check (no code changes yet)

Run through these and note what you see. Then we can fix once we know what’s going on.

---

## 1. How is the project configured in Vercel?

- Vercel dashboard → your **github-data-app** project → **Settings**.
- **General**: Is there a **Framework Preset**? (e.g. “Other”, “Vite”, “Next.js”.) If it’s set to something like “Next.js” or a static site, Vercel may not be running your Python app at all.
- **Build & Development**:
  - **Build Command**: Is it blank (auto) or something custom?
  - **Output Directory**: Is it set? (Python/FastAPI often use no output dir.)
- **Root Directory**: Is it blank (repo root) or a subfolder?

Note: Framework Preset, Build Command, Output Directory, Root Directory.

---

## 2. What did the last deploy actually build?

- **Deployments** → open the **latest deployment** (the one that gives 404).
- Check the **Build** step (logs): does it mention **Python** or **FastAPI**? Or does it look like a static/Node build?
- If there is a **Functions** (or “Serverless Functions”) section for that deployment, open it. You should see one or more functions with paths like:
  - `/api/...`
  - or something else
- Write down the **exact path(s)** of any function(s) (e.g. `/api/app/index`, `/api`, etc.).

---

## 3. Where does the app actually respond? (browser)

Use your **production URL** (e.g. `https://github-data-app.vercel.app`). Try in order:

1. `https://<your-app>.vercel.app/`  
   → You said this gives **404**.
2. `https://<your-app>.vercel.app/api`  
   → 404, 500, or something else? If you get JSON or HTML (not 404), the function is under `/api`.
3. `https://<your-app>.vercel.app/api/`  
   → Same question.
4. `https://<your-app>.vercel.app/api/health`  
   → 404 or JSON (e.g. `{"status":"ok",...}`)?
5. If you have an `app` folder: `https://<your-app>.vercel.app/api/app/index`  
   → 404 or something else?

Note for each: **200 + body**, **404**, **500**, or **other** (and what you see).

---

## 4. One thing that often causes 404

If **Framework Preset** is **not** “Other” (or similar “backend” style), Vercel may be building a **frontend** (e.g. static or Next) and **never** running your Python code. Then every request (including `/` and `/api/...`) can end up as 404 because there is no Python function in the build.

---

## What to send back

Reply with:

- Framework Preset (and Build Command / Output Dir if set).
- Whether the build logs mention Python/FastAPI.
- The **exact** function path(s) from the deployment (e.g. `/api/app/index`).
- For each test URL above: status (200/404/500) and a one-line description (e.g. “404”, “JSON with status”, “HTML page”).

With that we can say exactly why you get 404 and what to change in one go (config and/or code).
