# GRO840

## Environment setup

> [!IMPORTANT]
> Python (> 3.12.X) is required to be installed and added to `PATH`.

Windows:

```ps1
git clone --recurse-submodules git@github.com:MarcOlivierFecteau/gro640.git .\gro640\
cd .\gro640\
python -m venv .venv
.venv\Scripts\Activate.ps1
pip3 install numpy matplotlib scipy
pip3 install .\include\pyro\
```
