# sill_analysis

### Code Execution

```
# Please specify how to execute your code here

1.Use pca_sill.py input two sill data ,the out sill pca heatmap and csv
2.Use pca_br.py input two sill data and six bookroll data ,the output sill pca heatmap and csv
3.Use correlation.py requires input two sill data and six bookroll data, the output is corr csv and heatmap
4.Use correlation_pca.py input pca_sill.py and pca_br.py output ,the output corr pca heatmap and csv

Example input ./data 
Example Output in ./images and ./outputs

```

### Repo Structure

* The folder follow the structure as follow, please give meanful name for each file and folder:
```
    .
    ├── ...
    ├── images                  # images that contained experiment results
    │   ├── image_01.png        # example image 01
    │   ├── image_02.png        # example image 02
    │   └── ...                 # Other images
    ├── outputs                 # output CSV files
    │   ├── output_01.csv       # example csv 01
    │   ├── output_02.csv       # example csv 02
    │   └── ...                 # Other outputs
    ├── data                    # raw data for the current experiment
    ├── ├── data_01.csv         # condition detection model (CDM)
    ├── ├── data_02.csv         # anomaly detection model (ADM)
    │   └── ...                 # Other datasets
    ├── correlation.py          # correlation analysis between SILL and BookRoll datasets
    ├── correlation_pca.py      # correlation analysis between SILL and BookRoll datasets after pca
    ├── pca_sill.py             # PCA analysis for SILL dataset
    ├── pca_br.py               # PCA analysis for BookRoll dataset
    ├── can.py                  # CAN analysis between SILL and BookRoll datasets
    └── ...
```
