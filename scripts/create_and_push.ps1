# PowerShell Script to Initialize Git and Push to GitHub
# Assistive Sign Language Converter - GitHub Upload Helper

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GitHub Upload Helper" -ForegroundColor Cyan
Write-Host "Assistive Sign Language Converter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = git --version 2>&1
    Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "After installation, restart PowerShell and run this script again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Navigate to project directory
$projectPath = "D:\Assitive Sign Language Convertor"
if (Test-Path $projectPath) {
    Set-Location $projectPath
    Write-Host "✓ Navigated to project directory" -ForegroundColor Green
} else {
    Write-Host "✗ Project directory not found: $projectPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if .git exists
if (Test-Path ".git") {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
    $reinit = Read-Host "Do you want to reinitialize? (y/N)"
    if ($reinit -eq "y" -or $reinit -eq "Y") {
        Remove-Item -Recurse -Force .git
        Write-Host "Removed existing .git folder" -ForegroundColor Yellow
    } else {
        Write-Host "Using existing repository" -ForegroundColor Yellow
    }
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to initialize Git repository" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Configure Git (if not already configured)
Write-Host "Checking Git configuration..." -ForegroundColor Yellow
$userName = git config --global user.name
$userEmail = git config --global user.email

if (-not $userName) {
    Write-Host "Git user name not configured" -ForegroundColor Yellow
    $name = Read-Host "Enter your name (for Git commits)"
    git config --global user.name $name
    Write-Host "✓ Git user name configured" -ForegroundColor Green
} else {
    Write-Host "✓ Git user name: $userName" -ForegroundColor Green
}

if (-not $userEmail) {
    Write-Host "Git user email not configured" -ForegroundColor Yellow
    $email = Read-Host "Enter your email (for Git commits)"
    git config --global user.email $email
    Write-Host "✓ Git user email configured" -ForegroundColor Green
} else {
    Write-Host "✓ Git user email: $userEmail" -ForegroundColor Green
}

Write-Host ""

# Check if remote exists
$remoteExists = git remote get-url origin 2>$null
if ($remoteExists) {
    Write-Host "Remote 'origin' already exists: $remoteExists" -ForegroundColor Yellow
    $changeRemote = Read-Host "Do you want to change it? (y/N)"
    if ($changeRemote -eq "y" -or $changeRemote -eq "Y") {
        git remote remove origin
        Write-Host "Removed existing remote" -ForegroundColor Yellow
    } else {
        Write-Host "Using existing remote" -ForegroundColor Yellow
        $skipRemote = $true
    }
}

if (-not $skipRemote) {
    Write-Host ""
    Write-Host "GitHub Repository Setup" -ForegroundColor Cyan
    Write-Host "----------------------" -ForegroundColor Cyan
    Write-Host "Repository: https://github.com/PramodGunarathna/assistive-sign-language-converter" -ForegroundColor Green
    Write-Host ""
    
    $useDefault = Read-Host "Use default repository? (Y/n)"
    if ($useDefault -eq "n" -or $useDefault -eq "N") {
        $githubUsername = Read-Host "Enter your GitHub username"
        $repoName = Read-Host "Enter your repository name"
        $remoteUrl = "https://github.com/$githubUsername/$repoName.git"
    } else {
        $remoteUrl = "https://github.com/PramodGunarathna/assistive-sign-language-converter.git"
    }
    Write-Host ""
    Write-Host "Adding remote: $remoteUrl" -ForegroundColor Yellow
    git remote add origin $remoteUrl
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Remote added successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to add remote" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Add all files
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Files added successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to add files" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Show status
Write-Host "Repository Status:" -ForegroundColor Cyan
git status --short

Write-Host ""

# Create commit
$commitMessage = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Initial commit: Assistive Sign Language Converter"
}

Write-Host ""
Write-Host "Creating commit..." -ForegroundColor Yellow
git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Commit created successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create commit" -ForegroundColor Red
    Write-Host "Note: If you see 'nothing to commit', all files are already committed." -ForegroundColor Yellow
}

Write-Host ""

# Set branch to main
Write-Host "Setting branch to 'main'..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Branch set to 'main'" -ForegroundColor Green

Write-Host ""

# Push to GitHub
Write-Host "Ready to push to GitHub!" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan
Write-Host "You will be prompted for your GitHub credentials:" -ForegroundColor Yellow
Write-Host "  - Username: Your GitHub username" -ForegroundColor Yellow
Write-Host "  - Password: Use a Personal Access Token (not your password)" -ForegroundColor Yellow
Write-Host ""
Write-Host "To create a Personal Access Token:" -ForegroundColor Yellow
Write-Host "  1. Go to: https://github.com/settings/tokens" -ForegroundColor Yellow
Write-Host "  2. Generate new token (classic)" -ForegroundColor Yellow
Write-Host "  3. Select 'repo' scope" -ForegroundColor Yellow
Write-Host "  4. Copy and use the token as password" -ForegroundColor Yellow
Write-Host ""

$push = Read-Host "Do you want to push now? (Y/n)"
if ($push -ne "n" -and $push -ne "N") {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        $remoteUrl = git remote get-url origin
        $repoUrl = $remoteUrl -replace '\.git$', ''
        Write-Host "Your repository is available at:" -ForegroundColor Cyan
        Write-Host $repoUrl -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "✗ Push failed. Please check:" -ForegroundColor Red
        Write-Host "  1. Your GitHub credentials" -ForegroundColor Yellow
        Write-Host "  2. Repository exists on GitHub" -ForegroundColor Yellow
        Write-Host "  3. You have push access to the repository" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "You can try pushing manually later with:" -ForegroundColor Yellow
        Write-Host "  git push -u origin main" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "Skipped push. You can push later with:" -ForegroundColor Yellow
    Write-Host "  git push -u origin main" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Script completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"

