# 🚀 GitHub Upload Guide

This guide will help you upload your Assistive Sign Language Converter project to GitHub.

## 📋 Prerequisites

### Step 1: Install Git

If Git is not installed on your system:

1. **Download Git for Windows:**
   - Visit: https://git-scm.com/download/win
   - Download the installer
   - Run the installer with default settings
   - **Important:** During installation, select "Git from the command line and also from 3rd-party software"

2. **Verify Installation:**
   - Open a new PowerShell or Command Prompt
   - Run: `git --version`
   - You should see something like: `git version 2.x.x`

### Step 2: Create a GitHub Account

If you don't have a GitHub account:
1. Visit: https://github.com/signup
2. Create your account
3. Verify your email address

### Step 3: Create a New Repository on GitHub

1. Log in to GitHub
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name:** `assistive-sign-language-converter` (or your preferred name)
   - **Description:** "Bridging Communication Gaps for Hearing and Speech Impaired Individuals"
   - **Visibility:** Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## 🔧 Upload Process

### Method 1: Using PowerShell Script (Recommended)

1. **Open PowerShell in your project directory:**
   ```powershell
   cd "D:\Assitive Sign Language Convertor"
   ```

2. **Run the upload script:**
   ```powershell
   .\scripts\create_and_push.ps1
   ```

3. **Follow the prompts:**
   - Enter your GitHub username
   - Enter your repository name
   - The script will guide you through the process

### Method 2: Manual Upload (Step-by-Step)

#### Step 1: Initialize Git Repository

```powershell
cd "D:\Assitive Sign Language Convertor"
git init
```

#### Step 2: Configure Git (First time only)

```powershell
git config --global user.name "Your Name"
git config --global user.email "pramodnadishka.l@gmail.com"
```

#### Step 3: Add All Files

```powershell
git add .
```

#### Step 4: Create Initial Commit

```powershell
git commit -m "Initial commit: Assistive Sign Language Converter"
```

#### Step 5: Add Remote Repository

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

#### Step 6: Push to GitHub

```powershell
git branch -M main
git push -u origin main
```

**Note:** You'll be prompted for your GitHub credentials:
- **Username:** Your GitHub username
- **Password:** Use a Personal Access Token (not your GitHub password)

### Creating a Personal Access Token

If you need to create a Personal Access Token:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Sign Language Converter")
4. Select scopes: Check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)
7. Use this token as your password when pushing

## 📝 Important Notes

### Files Excluded from Upload

The `.gitignore` file is configured to exclude:
- Large model files (`.pth`, `.pt`, `.h5`)
- Feature files (`.npy`)
- Python cache files (`__pycache__/`)
- Virtual environments
- Log files

### Large Files Warning

If you need to upload large model files:
1. Use **Git LFS** (Large File Storage)
2. Or host them separately and link in README

### Updating Your README

After creating the repository, update the project link in `README.md`:
- Line 370: Replace `yourusername` with your actual GitHub username
- Replace `sign-language-converter` with your actual repository name

## 🔄 Future Updates

To push future changes:

```powershell
git add .
git commit -m "Description of your changes"
git push
```

## ❓ Troubleshooting

### Issue: "fatal: not a git repository"
**Solution:** Make sure you're in the project directory and run `git init`

### Issue: "fatal: remote origin already exists"
**Solution:** Remove and re-add:
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Issue: "Authentication failed"
**Solution:** Use Personal Access Token instead of password

### Issue: "Large files detected"
**Solution:** Remove large files from commit or use Git LFS

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git LFS Documentation](https://git-lfs.github.com/)

---

**Need Help?** Check the script output or refer to GitHub's documentation.

