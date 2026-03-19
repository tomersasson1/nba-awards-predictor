# Manual GitHub Steps for Portfolio Setup

After pushing the NBA project to GitHub, complete these steps to maximize recruiter visibility.

---

## 1. Create the NBA repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. **Repository name:** `nba-awards-predictor`
3. **Description:** `End-to-end ML pipeline predicting NBA award voting (MVP, DPOY, ROTY, MIP, 6MOY, COTY) with PyTorch, web scraping, and Plotly Dash`
4. Choose **Public**
5. Do **not** initialize with README (you already have one)
6. Click **Create repository**

Then run in your NBA project folder:

```powershell
git remote add origin https://github.com/tomersasson1/nba-awards-predictor.git
git branch -M main
git push -u origin main
```

---

## 2. Add topics to both repos

### NBA Awards Predictor
On [github.com/tomersasson1/nba-awards-predictor](https://github.com/tomersasson1/nba-awards-predictor):
- Click the gear icon next to **About** (or "Add topics")
- Add: `machine-learning`, `pytorch`, `nba`, `data-science`, `web-scraping`, `plotly-dash`, `python`

### CineMatch
On [github.com/tomersasson1/cinematch](https://github.com/tomersasson1/cinematch):
- Add: `machine-learning`, `recommendation-system`, `collaborative-filtering`, `streamlit`, `python`, `data-science`

---

## 3. Fix CineMatch repo description (typo)

On the CineMatch repo, click the gear next to **About** and set:
- **Description:** `Movie & TV recommendation system with collaborative filtering, content-based models, and Streamlit UI`  
  (Fix "Recommandation" → "Recommendation" if it appears in the current description.)

---

## 4. Create your GitHub Profile README

1. Create a new repo named **exactly** `tomersasson1` (same as your username)
2. Set it to **Public**
3. Check **Add a README file**
4. Click **Create repository**
5. Open the new `README.md` and replace its contents with the content from [`docs/GITHUB_PROFILE_README.md`](GITHUB_PROFILE_README.md) in this project
6. Commit and push

This README will appear on your GitHub profile when recruiters visit [github.com/tomersasson1](https://github.com/tomersasson1).

---

## 5. Pin both repos to your profile

1. Go to [github.com/tomersasson1](https://github.com/tomersasson1)
2. Click **Customize your pins**
3. Select `nba-awards-predictor` and `cinematch`
4. Click **Save pins**

---

## 6. Fill in your profile metadata

1. Click **Edit profile** (or the pencil icon) on your profile
2. **Bio:** e.g. `Data Science student | Python, ML, PyTorch`
3. Optional: add your location, website, Twitter/LinkedIn

---

## 7. Add a dashboard screenshot to the NBA README (optional but recommended)

1. Run the NBA dashboard: `python -m src.dashboard.app`
2. Open `http://127.0.0.1:8050` in a browser
3. Go to the "2025-26 Predictions" tab
4. Take a screenshot
5. Save it as `docs/dashboard_screenshot.png` in the NBA project
6. Update the NBA README: change `<!-- ADD SCREENSHOT: docs/dashboard_screenshot.png -->` to:
   ```markdown
   ![Dashboard](docs/dashboard_screenshot.png)
   ```
7. Commit and push
