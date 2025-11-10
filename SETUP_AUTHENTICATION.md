# 🔐 GitHub Authentication Setup Guide

## Step 1: Create Personal Access Token

1. **Go to GitHub Token Settings:**
   - Visit: https://github.com/settings/tokens
   - Or: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate New Token:**
   - Click **"Generate new token (classic)"** button
   - You may be prompted to enter your GitHub password

3. **Configure Token:**
   - **Note:** Give it a descriptive name like "Assistive Sign Language Converter"
   - **Expiration:** Choose your preferred expiration (90 days, 1 year, or no expiration)
   - **Select scopes:** Check the following:
     - ✅ **`repo`** - Full control of private repositories
       - This includes: repo:status, repo_deployment, public_repo, repo:invite, security_events

4. **Generate and Copy:**
   - Click **"Generate token"** at the bottom
   - **IMPORTANT:** Copy the token immediately! It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - You won't be able to see it again after you leave the page

## Step 2: Use Token for Authentication

When you run `git push`, you'll be prompted:
- **Username:** `PramodGunarathna`
- **Password:** Paste your Personal Access Token (NOT your GitHub password)

## Step 3: Optional - Save Credentials

To avoid entering credentials every time, you can configure Git Credential Manager:

```powershell
git config --global credential.helper manager-core
```

This will securely store your credentials in Windows Credential Manager.

---

## Alternative: Use GitHub Desktop

If you prefer a GUI:
1. Download: https://desktop.github.com/
2. Sign in with your GitHub account
3. It handles authentication automatically

---

## Ready to Push?

Once you have your Personal Access Token, we can push the commits!


