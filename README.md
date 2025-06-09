# GRO640

## Environment setup

### Requirements

> [!IMPORTANT]
> Python (> 3.12.X) is required to be installed and added to `PATH`.

> [!IMPORTANT]
> Because `pyro` is installed from a local repository, `pip freeze` "hard-codes" the path to `pyro`.
> For the installation to work, you MUST edit the path in `requirements.txt` (instructions [here](#editing-the-path-to-pyro)).

> [!NOTE]
> It is recommended that you clone this project to a **local** storage (i.e. on your machine, NOT in a cloud storage) in order for the environment setup to work properly. Installing the required Python packages does NOT seem to work when `pyro` (and therefore, the project) is located on a cloud storage (e.g. OneDrive). Thanks to Simon Lamontagne for finding this *bug*!

### Command pipeline

Windows:

```ps1
git clone --branch master --recurse-submodules git@github.com:MarcOlivierFecteau/gro640.git .\gro640\
cd .\gro640\
python -m venv .venv
# Activate the virtual environment
.venv\Scripts\Activate.ps1
# TODO: Edit the path to `pyro` in requirements.txt BEFORE installing with `pip`
pip3 install --requirement requirements.txt
```

> [!NOTE]
> Every time you open a new terminal, you MUST activate the virtual environment.

### Editing the path to pyro

Open `requirements.txt` in a text editor, find the line that starts with `pyro`. Here's what the line should look like:

```txt
pyro @ file:///C:/Users/marco/Documents/gro640/include/pyro
```

You MUST replace `C:/Users/marco/Documents/` for the local path to the project:

```txt
pyro @ file:///path/to/gro840/include/pyro
```

Thanks to Simon Lamontagne for finding this requirement!

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
