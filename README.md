# GRO640

## Environment setup

> [!IMPORTANT]
> Python (> 3.12.X) is required to be installed and added to `PATH`.

Windows:

```ps1
git clone --recurse-submodules git@github.com:MarcOlivierFecteau/gro640.git .\gro640\
cd .\gro640\
python -m venv .venv
# Activate the virtual environment
.venv\Scripts\Activate.ps1
pip3 install -r requirements.txt
```

> [!NOTE]
> Every time you open a new terminal, you MUST activate the virtual environment.

## Make it your own

> [!IMPORTANT]
> BEFORE running the commands, you MUST create an empty repository on GitHub.

Windows:

```ps1
cd path/to/gro640/
Remove-Item -Recurse -Force .git\
git init --initial-branch <branch_name> .\
Rename-Item .\src\prob\dosg0801_fecm0701.py <cip1_cip2>.py
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:<your_github_username>/<remote_repo_name>.git
git push -u origin <branch_name>
```
