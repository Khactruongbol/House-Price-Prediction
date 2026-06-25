# Ke Hoach Thuc Hien Chuong Trinh Du Doan Gia Nha

## 1. Muc Tieu

Xay dung chuong trinh Python du doan gia nha theo bai toan supervised regression. Bien dau vao la cac dac trung cua can nha nhu dien tich, chat luong, nam xay dung, garage, khu vuc va kieu nha. Bien dau ra la `SalePrice`.

## 2. Nguon Du Lieu Kaggle

Dataset chinh:

- House Prices - Advanced Regression Techniques: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Dataset tham khao hoac thay the:

- Housing Prices Dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
- Ames Housing Dataset: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset
- Real Estate Price Prediction: https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction
- Bengaluru House Price Data: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Sau khi tai dataset chinh, dat file `train.csv` vao `data/raw/train.csv`. Neu chua co Kaggle credential, chuong trinh thu thap real Ames Housing data tu OpenML. Workflow train chinh khong dung sample data.

## 3. Cau Truc Du An

```text
House-Price-Prediction/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── KE_HOACH_DU_DOAN_GIA_NHA.md
├── models/
├── notebooks/
│   └── 99_house_price_workflow.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explain_prediction.py
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    └── test_prediction.py
```

## 4. Workflow Tong The

1. Data collection: lay dataset tu Kaggle hoac OpenML va luu vao `data/raw/`.
2. Noise filtering: xoa duplicate rows, cot missing qua cao, cot hang so va outlier bat thuong.
3. Label filtering: chi giu label `SalePrice` hop le, khong missing va lon hon 0.
4. Data labeling: tao nhan phan tich `PriceSegment`, `QualityLabel`, `AreaSegment`.
5. Figure generation: ve hinh tu real filtered data va luu vao `reports/figures/`.
6. EDA: ve histogram `SalePrice`, scatter `GrLivArea` vs `SalePrice`, heatmap tuong quan va nhan dien outlier.
7. Feature engineering: tao them `HouseAgeAtRemodel`, `AreaPerBedroom`, `GarageAreaPerCar` neu cot nguon ton tai.
8. Preprocessing: impute missing value, standardize numeric features, one-hot encode categorical features bang `ColumnTransformer`.
9. Model training: train Linear Regression, Ridge, Lasso, Random Forest va Gradient Boosting bang data da loc.
10. Hyperparameter tuning: dung cross-validation de tune Ridge, Lasso, Random Forest va Gradient Boosting.
11. Optimization check: so sanh baseline va tuned models bang RMSE, MAE, R2; chon model co RMSE thap nhat.
12. Model saving: luu pipeline tot nhat vao `models/house_price_pipeline.joblib`.
13. Inference: nhap thong tin nha moi, ap dung dung pipeline da train va tra ve gia du doan.
14. Interface: dung HTML + Flask trong `app_html.py` va `templates/index.html` de nhap feature va xem prediction.

## 5. Workflow Training

Input training la dataset co feature `X` va target `y = SalePrice`. Chuong trinh chia train/test, fit preprocessing tren train set, train tung model, du doan test set, tinh metric va luu artifact.

Cong thuc danh gia:

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
MAE = mean(abs(y_true - y_pred))
R2 = 1 - SS_res / SS_tot
```

## 6. Workflow Inference

Input inference la mot dong thong tin nha moi. Chuong trinh bo sung feature engineering, can chinh dung danh sach feature da train, nap pipeline da luu, predict `SalePrice`, va hien thi ket qua trong giao dien.

## 7. Notebook Chinh

Notebook `notebooks/99_house_price_workflow.ipynb` la trang workflow tap trung. Notebook nay khong tach thanh nhieu notebook nho, de nguoi hoc co the chay tu dau den cuoi va thay ro toan bo qua trinh. Cau truc notebook mo phong final notebook spam email: problem definition, success metrics, data collection, EDA, data quality, preprocessing, feature engineering, model training, model metrics, tuning, residual analysis, learning curve, model selection, demo predict, HTML UI va checklist.

## 8. Google Colab

Khi train tren Google Colab:

1. Upload `kaggle.json`.
2. Cai `kaggle`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `flask`.
3. Tai competition `house-prices-advanced-regression-techniques`.
4. Giai nen file va dat `train.csv` vao `data/raw/train.csv`.
5. Chay notebook tu dau den cuoi de sinh model, metrics va hinh anh.

Notebook da co cell Colab setup mau, chi can bo comment khi chay tren Colab.

## 9. Hinh Anh Bao Cao

- `saleprice_distribution.png`: phan phoi gia nha.
- `grlivarea_vs_saleprice.png`: scatter dien tich song va gia.
- `saleprice_correlation_heatmap.png`: heatmap tuong quan voi gia.
- `model_rmse_comparison.png`: so sanh RMSE cac model.
- `actual_vs_predicted_saleprice.png`: gia thuc te vs gia du doan.
- `residual_distribution.png`: phan phoi sai so.
- `residuals_vs_predicted.png`: residual theo gia du doan.
- `learning_curve_best_model.png`: learning curve cua model tot nhat.

## 10. Kiem Thu Va Tieu Chi Hoan Thanh

- Notebook parse duoc JSON.
- Dataset co cot `SalePrice`.
- Pipeline train xong tao file model trong `models/`.
- File metric xuat ra `reports/metrics/model_metrics.csv`.
- File tuning xuat ra `reports/metrics/model_tuning_report.csv`.
- Cac hinh EDA/model xuat ra `reports/figures/`.
- Ham prediction tra ve mot gia tri so duong.
- HTML app co the chay bang `python app_html.py`.
