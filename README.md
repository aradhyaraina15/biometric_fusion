## Setup
```bash
pip install torch torchvision opencv-python scikit-image numpy pytest
```
## Usage
```python
from fingerprint.fingerprint_module import FingerprintModule
# Classical mode
fm = FingerprintModule(use_cnn=False)
# CNN mode (train first)
fm = FingerprintModule(use_cnn=True)
# Enroll
fm.enroll("finger.png", "templates/user_001.pkl")
# Verify
result = fm.verify("probe.png", "templates/user_001.pkl")
print(result.score, result.match)
# Identify against whole database
result = fm.identify("probe.png", "templates/")
print(result["identity"])
```
## Running tests
```bash
python -m pytest tests/ -v
```