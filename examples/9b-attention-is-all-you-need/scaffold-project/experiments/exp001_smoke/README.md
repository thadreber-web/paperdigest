# Smoke Test Run (exp001_smoke)

## Purpose
This smoke test validates that the training pipeline executes without errors using simplified hyperparameters. It checks:
- Config loading works correctly
- Model instantiation succeeds
- Training loop initializes properly
- Tracking/logging functions operate

## Command to Run
```
python train.py --config configs/smoke.yaml
```

## Expected Behavior
- Model should instantiate with reduced dimensions (d_model=64, N=2, h=2)
- Training should complete without errors
- Results should be logged via tracking system
- Runtime should complete in seconds (not hours)
