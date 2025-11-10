# 🚀 Upload Project to GitHub Repository

Your repository is ready at: **https://github.com/PramodGunarathna/assistive-sign-language-converter**

## 📋 Method 1: Using Git Command Line (Recommended)

### Step 1: Install Git

1. **Download Git for Windows:**
   - Visit: https://git-scm.com/download/win
   - Download and run the installer
   - **Important:** During installation, select "Git from the command line and also from 3rd-party software"
   - Complete the installation

2. **Restart PowerShell** after installation

### Step 2: Open PowerShell in Project Directory

```powershell
cd "D:\Assitive Sign Language Convertor"
```

### Step 3: Initialize Git Repository

```powershell
git init
```

### Step 4: Configure Git (First time only)

```powershell
git config --global user.name "Pramod Gunarathna"
git config --global user.email "pramodnadishka.l@gmail.com"
```

### Step 5: Add All Files

```powershell
git add .
```

### Step 6: Create Initial Commit

```powershell
git commit -m "Initial commit: Assistive Sign Language Converter"
```

### Step 7: Add Remote Repository

```powershell
git remote add origin https://github.com/PramodGunarathna/assistive-sign-language-converter.git
```

### Step 8: Set Branch to Main

```powershell
git branch -M main
```

### Step 9: Push to GitHub

```powershell
git push -u origin main
```

**When prompted:**
- **Username:** `PramodGunarathna`
- **Password:** Use a **Personal Access Token** (NOT your GitHub password)

#### Create Personal Access Token:

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Give it a name: "Sign Language Converter"
4. Select scope: **`repo`** (full control of private repositories)
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing

---

## 📋 Method 2: Using GitHub Desktop (Easier)

### Step 1: Download GitHub Desktop

1. Visit: https://desktop.github.com/
2. Download and install GitHub Desktop

### Step 2: Sign in to GitHub

1. Open GitHub Desktop
2. Sign in with your GitHub account (`PramodGunarathna`)

### Step 3: Add Local Repository

1. Click **"File"** → **"Add Local Repository"**
2. Click **"Choose..."**
3. Navigate to: `D:\Assitive Sign Language Convertor`
4. Click **"Add Repository"**

### Step 4: Publish Repository

1. Click **"Publish repository"** button
2. Repository name: `assistive-sign-language-converter`
3. Description: "Bridging Communication Gaps for Hearing and Speech Impaired Individuals"
4. Make sure **"Keep this code private"** is unchecked (if you want it public)
5. Click **"Publish Repository"**

---

## 📋 Method 3: Using PowerShell Script (Automated)

After installing Git, you can use the automated script:

```powershell
cd "D:\Assitive Sign Language Convertor"
.\scripts\create_and_push.ps1
```

The script will guide you through the process automatically.

---

## ✅ After Uploading

Once your code is uploaded, verify by visiting:
**https://github.com/PramodGunarathna/assistive-sign-language-converter**

You should see all your files there!

---

## 🔄 Future Updates

To push future changes:

```powershell
git add .
git commit -m "Description of your changes"
git push
```

---

## ❓ Troubleshooting

### Issue: "fatal: not a git repository"
**Solution:** Make sure you're in the project directory and run `git init`

### Issue: "fatal: remote origin already exists"
**Solution:** 
```powershell
git remote remove origin
git remote add origin https://github.com/PramodGunarathna/assistive-sign-language-converter.git
```

### Issue: "Authentication failed"
**Solution:** Use Personal Access Token instead of password

### Issue: "Large files detected"
**Solution:** The `.gitignore` file already excludes large model files (`.pth`, `.pt`, `.h5`, `.npy`). If you need to upload them, use Git LFS.

---

## 📝 Notes

- Large files (model files, `.npy` files) are excluded by `.gitignore`
- Your README.md has been updated with the correct repository URL
- The repository is currently empty and ready for your code


