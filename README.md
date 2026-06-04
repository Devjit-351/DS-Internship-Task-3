# 📊 Task 3: Exploratory Data Analysis (EDA) Project
### Internship Task #3 — Data Science

---

## 📁 Project Structure
```
DS-Internship-Task-3/
├── eda_analysis.py            # Full EDA pipeline script
├── ecommerce_data.csv         # Dataset (1,200 transactions)
├── eda_dashboard.png          # 15-panel visual dashboard
└── README.md                  # This file
```

---

## 🎯 Objective
Perform Exploratory Data Analysis on an **E-Commerce Sales Dataset** to uncover patterns, trends, and key influencing factors using statistical summaries and visualizations.

---

## 🛠️ Tools & Libraries
| Library | Purpose |
|---|---|
| **Pandas / NumPy** | Data manipulation & statistics |
| **Matplotlib** | Charts, subplots, layout |
| **Seaborn** | Heatmaps, violin plots |
| **SciPy** | Statistical tests, KDE, regression |

---

## 📊 Dataset Overview
- **1,200 e-commerce transactions** across 13 features
- **6 product categories**: Electronics, Clothing, Home & Kitchen, Books, Sports, Beauty
- **4 regions**: North, South, East, West
- **5 payment methods**: Credit Card, UPI, Debit Card, Net Banking, COD

| Feature | Description |
|---|---|
| Category | Product category |
| Region | Customer region |
| Age / Gender | Customer demographics |
| Month | Month of purchase |
| Price | Unit price ($) |
| Quantity | Items ordered |
| Revenue | Total order value ($) |
| Rating | Customer rating (1–5) |
| Discount_Pct | Discount applied (%) |
| Delivery_Days | Days to deliver |
| Returned | Whether order was returned |
| Payment_Method | Mode of payment |

---

## 📈 Statistical Summary

| Metric | Price ($) | Revenue ($) | Rating | Discount (%) |
|---|---|---|---|---|
| Mean | 178.14 | 358.42 | 3.90 | 11.68% |
| Median | 107.03 | 189.20 | 3.90 | 10.00% |
| Std Dev | 166.67 | 443.62 | 0.51 | 8.48% |
| Min | 15.02 | 15.02 | 2.10 | 0% |
| Max | 629.90 | 2976.95 | 5.00 | 30% |

---

## 🔍 Key Correlations
- **Price ↔ Revenue**: Strong positive (r = 0.78) — higher price = higher revenue
- **Quantity ↔ Revenue**: Moderate positive (r = 0.48)
- **Discount ↔ Return Rate**: Positive (r = 0.12) — more discount → more returns
- **Rating ↔ Revenue**: Weak negative (r = -0.06) — price doesn't drive satisfaction

---

## 🧠 Key Insights

### 1. Revenue & Sales
- 📦 **Electronics** is the top revenue category despite not having the most orders
- 🌍 **North region** leads in total revenue
- 📅 **July–October** shows the highest monthly revenue (peak season)

### 2. Customer Behavior
- 👥 **26–45 age group** spends the most per order
- 💳 **Credit Card & UPI** are the most popular payment methods (58% combined)
- 🎯 Customers with **higher discounts** are more likely to return orders

### 3. Product Quality
- ⭐ **Books** has the highest average rating (4.2)
- ⚡ **Faster delivery (1–2 days)** correlates with better ratings
- 🔁 Overall **return rate is 5.2%** — healthy for e-commerce

---

## 📉 Visualizations (15 Charts)
1. Total Revenue by Category
2. Monthly Revenue Trend (with peak annotation)
3. Revenue Distribution (KDE + Histogram)
4. Category Share — Donut Chart
5. Avg Rating by Category
6. Revenue by Region
7. Feature Correlation Heatmap
8. Price vs Revenue Scatter (by Category + trend line)
9. Avg Revenue by Age Group
10. Discount % vs Return Rate
11. Orders by Payment Method
12. Revenue Heatmap: Gender × Category
13. Rating Distribution by Category (Violin Plot)
14. Rating vs Delivery Speed (Boxplot)
15. KPI Summary Panel

---

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scipy
python eda_analysis.py
```

Outputs: `eda_dashboard.png` and `ecommerce_data.csv`

---

## 🏆 Summary KPIs
| KPI | Value |
|---|---|
| Total Revenue | $430,103 |
| Total Orders | 1,200 |
| Avg Order Value | $358.42 |
| Avg Customer Rating | 3.90 / 5.0 |
| Return Rate | 5.2% |
| Top Category | Electronics |
| Top Region | North |
| Top Payment Method | Credit Card |
