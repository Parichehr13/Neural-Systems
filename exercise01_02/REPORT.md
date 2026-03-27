# Exercise 01 and 02 Report

## Folder contents
- `exercises01_02.py`: original Python code (copied here).
- `run_exercises01_02.py`: headless runner used to execute the script and save figures.
- `figures/`: generated plots.
- `figures_manifest.json`: list of all generated figure files.

## Execution
Command used:

```powershell
python .\run_exercises01_02.py
```

Result:
- `50` figures were generated and saved in `figures/` as `fig_001.png` ... `fig_050.png`.

## Exercise 1 (constant threshold)
### Input current
![Exercise 1 - Input current](figures/fig_001.png)

### Membrane potential
![Exercise 1 - Membrane potential](figures/fig_002.png)

### Spike train
![Exercise 1 - Spike train](figures/fig_003.png)

### Example trial (constant current)
![Exercise 1 - Trial example](figures/fig_004.png)

### Current-frequency curve
![Exercise 1 - I-F curve](figures/fig_026.png)

## Exercise 2 (variable threshold)
### Single-current response
![Exercise 2 - Single current response](figures/fig_027.png)

### Example trial (constant current)
![Exercise 2 - Trial example](figures/fig_028.png)

### Current-frequency curve
![Exercise 2 - I-F curve](figures/fig_050.png)

## Notes
- Trial-by-trial figures are included in `fig_004.png` to `fig_025.png` (Exercise 1) and `fig_028.png` to `fig_049.png` (Exercise 2).
- All images are saved as PNG with `dpi=200`.
