import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("   TASK 3: EXPLORATORY DATA ANALYSIS (EDA) PROJECT")
print("=" * 65)

# ============================================================
# STEP 1: CREATE DATASET — E-Commerce Sales Data
# ============================================================
np.random.seed(42)
n = 1200

categories   = np.random.choice(['Electronics','Clothing','Home & Kitchen','Books','Sports','Beauty'], n,
                                  p=[0.25, 0.22, 0.20, 0.13, 0.12, 0.08])
regions      = np.random.choice(['North','South','East','West'], n, p=[0.28,0.25,0.27,0.20])
age          = np.random.randint(18, 65, n)
gender       = np.random.choice(['Male','Female'], n, p=[0.52, 0.48])
months       = np.random.choice(range(1,13), n,
                p=[0.06,0.06,0.07,0.08,0.08,0.09,0.10,0.10,0.09,0.10,0.09,0.08])

# Price by category
price_map = {'Electronics':350,'Clothing':65,'Home & Kitchen':120,'Books':25,'Sports':85,'Beauty':45}
base_price = np.array([price_map[c] for c in categories])
price      = (base_price * np.random.uniform(0.6, 1.8, n)).round(2)

# Quantity
qty        = np.random.choice([1,2,3,4,5], n, p=[0.45,0.30,0.14,0.07,0.04])

# Revenue
revenue    = (price * qty).round(2)

# Ratings (influenced by category and price)
rating_base = np.where(categories=='Books', 4.2,
              np.where(categories=='Electronics', 3.9,
              np.where(categories=='Beauty', 4.1, 3.8)))
rating = np.clip(rating_base + np.random.normal(0, 0.5, n), 1.0, 5.0).round(1)

# Discount %
discount = np.random.choice([0,5,10,15,20,25,30], n, p=[0.20,0.15,0.20,0.18,0.15,0.08,0.04])

# Delivery days
delivery_days = np.random.randint(1, 10, n)

# Returned
return_prob = 0.05 + 0.10*(discount>=25) + 0.08*(rating<=2.5) - 0.05*(delivery_days<=2)
return_prob = np.clip(return_prob, 0.01, 0.35)
returned    = (np.random.rand(n) < return_prob).astype(int)

# Payment
payment = np.random.choice(['Credit Card','UPI','Debit Card','Net Banking','COD'], n,
                             p=[0.30,0.28,0.18,0.12,0.12])

df = pd.DataFrame({
    'Category': categories, 'Region': regions,
    'Age': age, 'Gender': gender, 'Month': months,
    'Price': price, 'Quantity': qty, 'Revenue': revenue,
    'Rating': rating, 'Discount_Pct': discount,
    'Delivery_Days': delivery_days, 'Returned': returned,
    'Payment_Method': payment
})

df['Month_Name'] = pd.Categorical(
    df['Month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                     7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}),
    categories=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)
df['Age_Group'] = pd.cut(df['Age'], bins=[17,25,35,45,55,65],
                          labels=['18-25','26-35','36-45','46-55','56-65'])

print(f"\nDataset Shape: {df.shape}")
print(f"\nStatistical Summary:\n{df[['Price','Quantity','Revenue','Rating','Discount_Pct','Delivery_Days']].describe().round(2).to_string()}")
print(f"\nCategory Distribution:\n{df['Category'].value_counts().to_string()}")
print(f"\nCorrelation Matrix:\n{df[['Price','Quantity','Revenue','Rating','Discount_Pct','Delivery_Days','Returned']].corr().round(3).to_string()}")

# ============================================================
# STEP 2: VISUALIZATIONS — 15 CHARTS
# ============================================================
C = {'p':'#1E3A5F','s':'#2874A6','a':'#7FB3D3','l':'#D6EAF8','bg':'#F4F8FB',
     'dk':'#0D1F2D','r':'#C0392B','o':'#E67E22','g':'#1A7A4A','pu':'#6C3483',
     'teal':'#148F77','gold':'#B7950B','gr':'#95A5A6'}
CMAP_CAT = [C['p'],C['s'],C['g'],C['o'],C['pu'],C['teal']]

plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})

fig = plt.figure(figsize=(22, 30))
fig.patch.set_facecolor(C['bg'])
fig.text(0.5, 0.992, '📊 Exploratory Data Analysis — E-Commerce Sales Dataset',
         fontsize=27, fontweight='bold', ha='center', va='top', color=C['dk'])
fig.text(0.5, 0.977, f'1,200 transactions  |  6 product categories  |  4 regions  |  Statistical summaries, correlations & trend analysis',
         fontsize=12, ha='center', va='top', color=C['s'])

gs = GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35,
              top=0.970, bottom=0.025, left=0.06, right=0.97)

# ── 1. Revenue by Category (horizontal bar) ──────────────────
ax = fig.add_subplot(gs[0, 0])
rev_cat = df.groupby('Category')['Revenue'].sum().sort_values()
bars = ax.barh(rev_cat.index, rev_cat.values/1000, color=CMAP_CAT, edgecolor='white')
for b, v in zip(bars, rev_cat.values/1000):
    ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f'${v:.1f}K', va='center', fontsize=8.5)
ax.set_title('Total Revenue by Category', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_xlabel('Revenue ($K)'); ax.set_facecolor('#fff')

# ── 2. Monthly Revenue Trend ─────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
monthly = df.groupby('Month_Name', observed=True)['Revenue'].sum() / 1000
ax.plot(monthly.index, monthly.values, color=C['s'], lw=2.5, marker='o', markersize=6, zorder=3)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color=C['s'])
ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly.index, rotation=45, fontsize=8)
ax.set_title('Monthly Revenue Trend', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_ylabel('Revenue ($K)'); ax.set_facecolor('#fff')
peak_idx = monthly.values.argmax()
ax.annotate(f'Peak\n${monthly.values[peak_idx]:.1f}K', xy=(peak_idx, monthly.values[peak_idx]),
            xytext=(peak_idx+1, monthly.values[peak_idx]+2),
            arrowprops=dict(arrowstyle='->', color=C['r']), fontsize=7.5, color=C['r'])

# ── 3. Distribution of Revenue (histogram + KDE) ─────────────
ax = fig.add_subplot(gs[0, 2])
ax.hist(df['Revenue'], bins=35, color=C['a'], edgecolor='white', linewidth=0.5, alpha=0.85, density=True)
kde_x = np.linspace(df['Revenue'].min(), df['Revenue'].max(), 300)
kde = stats.gaussian_kde(df['Revenue'])
ax.plot(kde_x, kde(kde_x), color=C['p'], lw=2.5, label='KDE')
ax.axvline(df['Revenue'].mean(), color=C['r'], ls='--', lw=1.8, label=f"Mean: ${df['Revenue'].mean():.0f}")
ax.axvline(df['Revenue'].median(), color=C['o'], ls='--', lw=1.8, label=f"Median: ${df['Revenue'].median():.0f}")
ax.set_title('Revenue Distribution', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_xlabel('Revenue ($)'); ax.legend(fontsize=8); ax.set_facecolor('#fff')

# ── 4. Category Share (Donut) ────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
cat_counts = df['Category'].value_counts()
wedges, texts, autotexts = ax.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
                                   colors=CMAP_CAT, startangle=90, pctdistance=0.78,
                                   wedgeprops={'edgecolor':'white','linewidth':2})
centre_circle = plt.Circle((0,0), 0.55, fc='white')
ax.add_artist(centre_circle)
for at in autotexts: at.set_fontsize(8)
ax.set_title('Category Share (Orders)', fontsize=11, fontweight='bold', color=C['dk'])

# ── 5. Avg Rating by Category ────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
avg_rating = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
bars5 = ax.bar(avg_rating.index, avg_rating.values, color=CMAP_CAT[:len(avg_rating)], edgecolor='white')
ax.set_ylim(0, 5.5)
for b, v in zip(bars5, avg_rating.values):
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.05, f'{v:.2f}⭐',
            ha='center', fontsize=8.5, fontweight='bold')
ax.axhline(df['Rating'].mean(), color=C['r'], ls='--', lw=1.5, label=f'Overall Avg: {df["Rating"].mean():.2f}')
ax.set_title('Avg Rating by Category', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_ylabel('Rating'); ax.set_xticklabels(avg_rating.index, rotation=25, ha='right', fontsize=8)
ax.legend(fontsize=8); ax.set_facecolor('#fff')

# ── 6. Revenue by Region ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
rev_region = df.groupby('Region')['Revenue'].sum() / 1000
colors6 = [C['p'], C['g'], C['o'], C['pu']]
bars6 = ax.bar(rev_region.index, rev_region.values, color=colors6, edgecolor='white')
for b, v in zip(bars6, rev_region.values):
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.5, f'${v:.1f}K',
            ha='center', fontsize=9, fontweight='bold')
ax.set_title('Total Revenue by Region', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_ylabel('Revenue ($K)'); ax.set_facecolor('#fff')

# ── 7. Correlation Heatmap ───────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
num_cols = ['Price','Quantity','Revenue','Rating','Discount_Pct','Delivery_Days','Returned']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, linecolor='white', annot_kws={'size':7.5},
            xticklabels=[c.replace('_','\n') for c in num_cols],
            yticklabels=[c.replace('_','\n') for c in num_cols])
ax.set_title('Feature Correlation Heatmap', fontsize=11, fontweight='bold', color=C['dk'])
ax.tick_params(labelsize=7)

# ── 8. Price vs Revenue Scatter ──────────────────────────────
ax = fig.add_subplot(gs[2, 1])
cat_list = df['Category'].unique()
for cat, col in zip(cat_list, CMAP_CAT):
    mask8 = df['Category'] == cat
    ax.scatter(df[mask8]['Price'], df[mask8]['Revenue'], alpha=0.35, s=18,
               color=col, label=cat, edgecolors='none')
# Trend line
m, b_val, r, p_val, _ = stats.linregress(df['Price'], df['Revenue'])
x_line = np.linspace(df['Price'].min(), df['Price'].max(), 200)
ax.plot(x_line, m*x_line+b_val, color=C['dk'], lw=2, ls='--', label=f'Trend (r={r:.2f})')
ax.set_xlabel('Unit Price ($)'); ax.set_ylabel('Revenue ($)')
ax.set_title('Price vs Revenue by Category', fontsize=11, fontweight='bold', color=C['dk'])
ax.legend(fontsize=6.5, ncol=2); ax.set_facecolor('#fff')

# ── 9. Age Group vs Avg Revenue ──────────────────────────────
ax = fig.add_subplot(gs[2, 2])
age_rev = df.groupby('Age_Group', observed=True)['Revenue'].mean()
bars9 = ax.bar(age_rev.index, age_rev.values, color=C['teal'], alpha=0.85, edgecolor='white')
for b, v in zip(bars9, age_rev.values):
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+1, f'${v:.0f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_title('Avg Revenue by Age Group', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_ylabel('Avg Revenue ($)'); ax.set_xlabel('Age Group'); ax.set_facecolor('#fff')

# ── 10. Discount vs Return Rate ──────────────────────────────
ax = fig.add_subplot(gs[3, 0])
disc_return = df.groupby('Discount_Pct')['Returned'].mean() * 100
ax.plot(disc_return.index, disc_return.values, color=C['r'], lw=2.5, marker='o', markersize=7)
ax.fill_between(disc_return.index, disc_return.values, alpha=0.15, color=C['r'])
ax.set_xlabel('Discount %'); ax.set_ylabel('Return Rate (%)')
ax.set_title('Discount % vs Return Rate', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_facecolor('#fff')

# ── 11. Payment Method Distribution ─────────────────────────
ax = fig.add_subplot(gs[3, 1])
pay_counts = df['Payment_Method'].value_counts()
colors11 = [C['p'],C['s'],C['g'],C['o'],C['pu']]
bars11 = ax.barh(pay_counts.index, pay_counts.values, color=colors11, edgecolor='white')
for b, v in zip(bars11, pay_counts.values):
    ax.text(b.get_width()+3, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=9)
ax.set_title('Orders by Payment Method', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_xlabel('Order Count'); ax.set_facecolor('#fff')

# ── 12. Gender × Category Revenue Heatmap ───────────────────
ax = fig.add_subplot(gs[3, 2])
pivot = df.pivot_table(values='Revenue', index='Gender', columns='Category', aggfunc='sum') / 1000
sns.heatmap(pivot, ax=ax, annot=True, fmt='.1f', cmap='Blues', linewidths=0.5,
            linecolor='white', annot_kws={'size':8.5})
ax.set_title('Revenue ($K): Gender × Category', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(labelsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

# ── 13. Rating Distribution (violin) ────────────────────────
ax = fig.add_subplot(gs[4, 0])
cat_order = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).index
data_violin = [df[df['Category']==c]['Rating'].values for c in cat_order]
vp = ax.violinplot(data_violin, positions=range(len(cat_order)), showmedians=True, showmeans=False)
for pc, col in zip(vp['bodies'], CMAP_CAT):
    pc.set_facecolor(col); pc.set_alpha(0.7)
vp['cmedians'].set_color(C['dk']); vp['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(cat_order)))
ax.set_xticklabels(cat_order, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Rating'); ax.set_title('Rating Distribution by Category (Violin)', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_facecolor('#fff')

# ── 14. Delivery Days vs Rating (Boxplot) ───────────────────
ax = fig.add_subplot(gs[4, 1])
df['Delivery_Group'] = pd.cut(df['Delivery_Days'], bins=[0,2,4,6,9],
                               labels=['1-2 days','3-4 days','5-6 days','7-9 days'])
dgroups = ['1-2 days','3-4 days','5-6 days','7-9 days']
data_box = [df[df['Delivery_Group']==g]['Rating'].dropna().values for g in dgroups]
bp = ax.boxplot(data_box, labels=dgroups, patch_artist=True,
                boxprops=dict(facecolor=C['a'], alpha=0.75),
                medianprops=dict(color=C['dk'], linewidth=2),
                whiskerprops=dict(color=C['s']), capprops=dict(color=C['s']))
ax.set_title('Rating vs Delivery Speed', fontsize=11, fontweight='bold', color=C['dk'])
ax.set_ylabel('Rating'); ax.set_xlabel('Delivery Time'); ax.set_facecolor('#fff')

# ── 15. KPI Summary Box ──────────────────────────────────────
ax = fig.add_subplot(gs[4, 2])
ax.axis('off')
ax.add_patch(plt.Rectangle((0,0),1,1, transform=ax.transAxes,
             fill=True, facecolor='#EAF2FB', edgecolor=C['s'], linewidth=2))
ax.set_title('📌 EDA Summary — Key Insights', fontsize=11, fontweight='bold', color=C['dk'], pad=10)
top_cat = df.groupby('Category')['Revenue'].sum().idxmax()
top_reg = df.groupby('Region')['Revenue'].sum().idxmax()
top_pay = df['Payment_Method'].mode()[0]
kpis = [
    ('Total Revenue',     f"${df['Revenue'].sum():,.0f}"),
    ('Total Orders',      f"{len(df):,}"),
    ('Avg Order Value',   f"${df['Revenue'].mean():.2f}"),
    ('Avg Rating',        f"{df['Rating'].mean():.2f} / 5.0"),
    ('Return Rate',       f"{df['Returned'].mean()*100:.1f}%"),
    ('Top Category',      top_cat),
    ('Top Region',        top_reg),
    ('Top Payment',       top_pay),
]
for i,(lbl,val) in enumerate(kpis):
    y_pos = 0.90 - i*0.112
    ax.text(0.04, y_pos, f"▸ {lbl}:", transform=ax.transAxes,
            fontsize=9, color=C['s'], fontweight='bold')
    ax.text(0.97, y_pos, str(val), transform=ax.transAxes,
            fontsize=9.5, color=C['dk'], ha='right', fontweight='bold')

plt.savefig('/home/claude/task3_eda_dashboard.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
df.drop(columns=['Month_Name','Age_Group','Delivery_Group']).to_csv('/home/claude/task3_ecommerce_data.csv', index=False)

print("\n\n✅ Dashboard saved: task3_eda_dashboard.png")
print("✅ Dataset saved  : task3_ecommerce_data.csv")

# Print key stats
print(f"\n{'─'*50}")
print("KEY FINDINGS")
print(f"{'─'*50}")
print(f"Total Revenue    : ${df['Revenue'].sum():,.0f}")
print(f"Avg Order Value  : ${df['Revenue'].mean():.2f}")
print(f"Top Category     : {top_cat}")
print(f"Top Region       : {top_reg}")
print(f"Avg Rating       : {df['Rating'].mean():.2f}")
print(f"Return Rate      : {df['Returned'].mean()*100:.1f}%")
print(f"Top Payment      : {top_pay}")
