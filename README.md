# ECG_CNN

### Download Dataset
```bash
mkdir mit-bih-arrhythmia-database && cd mit-bih-arrhythmia-database && url=https://physionet.org/physiobank/database/mitdb/ && for i in {100..234}; do for ext in 'hea' 'dat' 'atr'; do wget "$url/$i.$ext"; done; done

```
