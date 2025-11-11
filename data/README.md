### Datasets

To generate mutant structures and prepare the processed datasets for SKEMPI v2.0, CR6261, HER2, and S285, respectively, execute the following commands from the code directory:

```python skempi_parallel.py --reset --subset skempi_v2
python skempi_parallel.py --reset --subset skempi_v2
python skempi_parallel.py --reset --subset CR6261
python skempi_parallel.py --reset --subset HER2
python skempi_parallel.py --reset --subset S285
```

or preprocessed dataset files for CR6261 and HER2  are available at [CR6261_cache](https://drive.google.com/file/d/1FgLHlM3xF847Kh8x3oFFtgJdzKDfTeLo/view?usp=sharing) and [HER2_cache](https://drive.google.com/file/d/1jHHuot3FjieTOBu293XP2uUk7xh_gD7B/view?usp=sharing), then place CR6261_cache and HER2_cache into the **SKEMPI2** directory, respectively.
