# Automation script to push project to GitHub and deploy to Vercel
param(
    [string]$RepoUrl = ""
)

$env:PATH += ";C:\Users\ankit\.gemini\antigravity\scratch\mingit\cmd;C:\Users\ankit\.gemini\antigravity\scratch\gh\bin"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   AI-Based Face Recognition Attendance System" -ForegroundColor Cyan
Write-Host "   GitHub & Vercel Automated Deployment Assistant" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if GitHub remote already exists
$existingRemote = git remote get-url origin 2>$null

if (-not $existingRemote) {
    if ($RepoUrl -ne "") {
        Write-Host "[+] Adding remote origin: $RepoUrl" -ForegroundColor Green
        git remote add origin $RepoUrl
    } else {
        Write-Host "[?] No remote 'origin' is set." -ForegroundColor Yellow
        Write-Host "Checking GitHub CLI (gh) authentication..." -ForegroundColor Yellow
        $ghStatus = gh auth status 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[+] GitHub CLI is authenticated!" -ForegroundColor Green
            Write-Host "[+] Creating GitHub repository automatically..." -ForegroundColor Cyan
            gh repo create face-recognition-attendance --public --source=. --remote=origin --push
            Write-Host "[+] Successfully created and pushed to GitHub!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "----------------------------------------------------------" -ForegroundColor Yellow
            Write-Host "Action Required: Choose one of the following methods:" -ForegroundColor Yellow
            Write-Host "Method 1: Log in with GitHub CLI:" -ForegroundColor White
            Write-Host "   gh auth login" -ForegroundColor Cyan
            Write-Host "   gh repo create face-attendance-system --public --source=. --remote=origin --push" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Method 2: If you created an empty repo on https://github.com/new:" -ForegroundColor White
            Write-Host "   git remote add origin https://github.com/<your-username>/<your-repo>.git" -ForegroundColor Cyan
            Write-Host "   git push -u origin main" -ForegroundColor Cyan
            Write-Host "----------------------------------------------------------" -ForegroundColor Yellow
            return
        }
    }
}

# 2. Push to GitHub
Write-Host "[*] Pushing latest code to GitHub..." -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[+] SUCCESS! Your code is live on GitHub." -ForegroundColor Green
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Magenta
    Write-Host "   NEXT STEP: Automatic Vercel Live Deployment (1 Click)" -ForegroundColor Magenta
    Write-Host "==========================================================" -ForegroundColor Magenta
    Write-Host "1. Visit https://vercel.com/new" -ForegroundColor White
    Write-Host "2. Click 'Import' on your new GitHub repository." -ForegroundColor White
    Write-Host "3. Click 'Deploy' (Vercel automatically detects vercel.json & Python WSGI)." -ForegroundColor White
    Write-Host "4. Any future git push to this repo will deploy automatically!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Magenta
} else {
    Write-Host "[-] Push failed. Please check your GitHub credentials or repository URL." -ForegroundColor Red
}
