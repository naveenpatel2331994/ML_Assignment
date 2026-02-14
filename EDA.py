"""
Exploratory Data Analysis (EDA) for Recomart Product Catalog Dataset
=====================================================================
This script performs comprehensive EDA on the product catalog data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load the dataset
df_copy = pd.read_csv('/Users/purushottampandey/Documents/DM4ML/recomart_products_data/recomart_product_catalog.csv')
df = df_copy.copy()

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - RECOMART PRODUCT CATALOG")
print("=" * 80)

# ============================================================================
# 1. BASIC INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("1. BASIC DATASET INFORMATION")
print("=" * 80)

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")

# ============================================================================
# 2. STATISTICAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("2. STATISTICAL SUMMARY")
print("=" * 80)

# Numerical columns summary
numerical_cols = ['base_price', 'discount_percent', 'monthly_sales_volume', 
                  'avg_rating', 'return_rate_percent', 'profit_margin_percent', 
                  'shelf_life_days']

print("\n📊 Numerical Columns Summary:")
print(df[numerical_cols].describe().round(2))

# Categorical columns summary
categorical_cols = ['super_category', 'category', 'brand', 'is_perishable']
print("\n📁 Categorical Columns Summary:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Top 5 values:\n{df[col].value_counts().head()}")

# ============================================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. MISSING VALUE ANALYSIS")
print("=" * 80)

missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing %': missing_percent
})
print(f"\nMissing Values:\n{missing_df[missing_df['Missing Count'] > 0] if missing_df['Missing Count'].sum() > 0 else 'No missing values found! ✅'}")

# ============================================================================
# 4. CATEGORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. CATEGORY ANALYSIS")
print("=" * 80)

print("\n🛒 Super Category Distribution:")
super_cat_stats = df.groupby('super_category').agg({
    'product_id': 'count',
    'base_price': 'mean',
    'monthly_sales_volume': 'mean',
    'avg_rating': 'mean',
    'profit_margin_percent': 'mean'
}).round(2)
super_cat_stats.columns = ['Product Count', 'Avg Price', 'Avg Sales', 'Avg Rating', 'Avg Profit Margin']
print(super_cat_stats.sort_values('Product Count', ascending=False))

print("\n📦 Category Distribution:")
print(df['category'].value_counts())

# ============================================================================
# 5. BRAND ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. BRAND ANALYSIS")
print("=" * 80)

print("\n🏷️ Top 15 Brands by Product Count:")
brand_counts = df['brand'].value_counts().head(15)
print(brand_counts)

print("\n🏆 Top 10 Brands by Average Monthly Sales:")
brand_sales = df.groupby('brand')['monthly_sales_volume'].mean().sort_values(ascending=False).head(10)
print(brand_sales.round(2))

print("\n⭐ Top 10 Brands by Average Rating:")
brand_rating = df.groupby('brand')['avg_rating'].mean().sort_values(ascending=False).head(10)
print(brand_rating.round(2))

# ============================================================================
# 6. PRICE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. PRICE ANALYSIS")
print("=" * 80)

print(f"\n💰 Price Statistics:")
print(f"  Min Price: ₹{df['base_price'].min()}")
print(f"  Max Price: ₹{df['base_price'].max()}")
print(f"  Mean Price: ₹{df['base_price'].mean():.2f}")
print(f"  Median Price: ₹{df['base_price'].median()}")
print(f"  Std Dev: ₹{df['base_price'].std():.2f}")

print("\n💸 Price Range by Super Category:")
price_by_category = df.groupby('super_category')['base_price'].agg(['min', 'max', 'mean']).round(2)
print(price_by_category)

# ============================================================================
# 7. SALES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. SALES ANALYSIS")
print("=" * 80)

print(f"\n📈 Sales Statistics:")
print(f"  Min Monthly Sales: {df['monthly_sales_volume'].min()}")
print(f"  Max Monthly Sales: {df['monthly_sales_volume'].max()}")
print(f"  Mean Monthly Sales: {df['monthly_sales_volume'].mean():.2f}")
print(f"  Median Monthly Sales: {df['monthly_sales_volume'].median()}")

print("\n🔥 Top 10 Products by Sales Volume:")
top_sales = df.nlargest(10, 'monthly_sales_volume')[['product_name', 'brand', 'super_category', 'monthly_sales_volume']]
print(top_sales.to_string(index=False))

# ============================================================================
# 8. RATING ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("8. RATING ANALYSIS")
print("=" * 80)

print(f"\n⭐ Rating Statistics:")
print(f"  Min Rating: {df['avg_rating'].min()}")
print(f"  Max Rating: {df['avg_rating'].max()}")
print(f"  Mean Rating: {df['avg_rating'].mean():.2f}")
print(f"  Median Rating: {df['avg_rating'].median()}")

print("\n📊 Rating Distribution:")
rating_bins = [0, 3, 3.5, 4, 4.5, 5]
rating_labels = ['1-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']
df['rating_range'] = pd.cut(df['avg_rating'], bins=rating_bins, labels=rating_labels)
print(df['rating_range'].value_counts().sort_index())

# ============================================================================
# 9. PROFIT MARGIN ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("9. PROFIT MARGIN ANALYSIS")
print("=" * 80)

print(f"\n💵 Profit Margin Statistics:")
print(f"  Min Profit Margin: {df['profit_margin_percent'].min():.2f}%")
print(f"  Max Profit Margin: {df['profit_margin_percent'].max():.2f}%")
print(f"  Mean Profit Margin: {df['profit_margin_percent'].mean():.2f}%")
print(f"  Median Profit Margin: {df['profit_margin_percent'].median():.2f}%")

print("\n📈 Profit Margin by Super Category:")
profit_by_cat = df.groupby('super_category')['profit_margin_percent'].agg(['mean', 'min', 'max']).round(2)
print(profit_by_cat)

# ============================================================================
# 10. RETURN RATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("10. RETURN RATE ANALYSIS")
print("=" * 80)

print(f"\n🔄 Return Rate Statistics:")
print(f"  Min Return Rate: {df['return_rate_percent'].min():.2f}%")
print(f"  Max Return Rate: {df['return_rate_percent'].max():.2f}%")
print(f"  Mean Return Rate: {df['return_rate_percent'].mean():.2f}%")

print("\n⚠️ Products with High Return Rate (>30%):")
high_return = df[df['return_rate_percent'] > 30][['product_name', 'brand', 'return_rate_percent', 'super_category']]
print(f"  Count: {len(high_return)} products")
print(high_return.head(10).to_string(index=False))

# ============================================================================
# 11. PERISHABLE ITEMS ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("11. PERISHABLE ITEMS ANALYSIS")
print("=" * 80)

print(f"\n🥛 Perishable vs Non-Perishable:")
print(df['is_perishable'].value_counts())

print("\n📊 Perishable Products by Super Category:")
perishable_analysis = df[df['is_perishable'] == 'Yes'].groupby('super_category').size()
print(perishable_analysis)

print("\n⏰ Shelf Life Statistics (for perishable items):")
perishable_df = df[df['is_perishable'] == 'Yes']
print(f"  Min Shelf Life: {perishable_df['shelf_life_days'].min()} days")
print(f"  Max Shelf Life: {perishable_df['shelf_life_days'].max()} days")
print(f"  Mean Shelf Life: {perishable_df['shelf_life_days'].mean():.2f} days")

# ============================================================================
# 12. DISCOUNT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("12. DISCOUNT ANALYSIS")
print("=" * 80)

print(f"\n🏷️ Discount Statistics:")
print(f"  Products with No Discount: {len(df[df['discount_percent'] == 0])}")
print(f"  Products with Discount: {len(df[df['discount_percent'] > 0])}")
print(f"  Max Discount: {df['discount_percent'].max()}%")
print(f"  Mean Discount: {df['discount_percent'].mean():.2f}%")

print("\n💰 Average Discount by Super Category:")
discount_by_cat = df.groupby('super_category')['discount_percent'].mean().round(2)
print(discount_by_cat)

# ============================================================================
# 13. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("13. CORRELATION ANALYSIS")
print("=" * 80)

correlation_cols = ['base_price', 'discount_percent', 'monthly_sales_volume', 
                    'avg_rating', 'return_rate_percent', 'profit_margin_percent']
correlation_matrix = df[correlation_cols].corr()
print("\n🔗 Correlation Matrix:")
print(correlation_matrix.round(3))

print("\n📌 Key Correlations:")
print(f"  Price vs Sales: {correlation_matrix.loc['base_price', 'monthly_sales_volume']:.3f}")
print(f"  Rating vs Sales: {correlation_matrix.loc['avg_rating', 'monthly_sales_volume']:.3f}")
print(f"  Return Rate vs Sales: {correlation_matrix.loc['return_rate_percent', 'monthly_sales_volume']:.3f}")
print(f"  Profit Margin vs Return Rate: {correlation_matrix.loc['profit_margin_percent', 'return_rate_percent']:.3f}")

# ============================================================================
# 14. LAUNCH DATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("14. LAUNCH DATE ANALYSIS")
print("=" * 80)

df['launch_date'] = pd.to_datetime(df['launch_date'])
df['launch_year'] = df['launch_date'].dt.year
df['launch_month'] = df['launch_date'].dt.month

print("\n📅 Products Launched by Year:")
print(df['launch_year'].value_counts().sort_index())

print("\n🆕 Products Launched in 2024:")
new_products_2024 = df[df['launch_year'] == 2024][['product_name', 'brand', 'launch_date', 'super_category']]
print(f"  Count: {len(new_products_2024)}")
print(new_products_2024.head(10).to_string(index=False))

# ============================================================================
# 15. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("15. KEY INSIGHTS SUMMARY")
print("=" * 80)

insights = f"""
📊 DATASET OVERVIEW:
   • Total Products: {df.shape[0]}
   • Total Brands: {df['brand'].nunique()}
   • Total Categories: {df['category'].nunique()}
   • Super Categories: {df['super_category'].nunique()}

💰 PRICING INSIGHTS:
   • Average Product Price: ₹{df['base_price'].mean():.2f}
   • Price Range: ₹{df['base_price'].min()} - ₹{df['base_price'].max()}
   • Most expensive category: {df.groupby('super_category')['base_price'].mean().idxmax()}

📈 SALES INSIGHTS:
   • Top selling category: {df.groupby('super_category')['monthly_sales_volume'].mean().idxmax()}
   • Average monthly sales: {df['monthly_sales_volume'].mean():.0f} units
   • Highest selling product: {df.loc[df['monthly_sales_volume'].idxmax(), 'product_name']}

⭐ RATING INSIGHTS:
   • Average product rating: {df['avg_rating'].mean():.2f}/5
   • Highest rated category: {df.groupby('super_category')['avg_rating'].mean().idxmax()}
   • Products with rating > 4.5: {len(df[df['avg_rating'] > 4.5])}

💵 PROFITABILITY:
   • Average profit margin: {df['profit_margin_percent'].mean():.2f}%
   • Most profitable category: {df.groupby('super_category')['profit_margin_percent'].mean().idxmax()}
   • Highest margin product: {df.loc[df['profit_margin_percent'].idxmax(), 'product_name']}

🔄 RETURN RATES:
   • Average return rate: {df['return_rate_percent'].mean():.2f}%
   • Category with highest returns: {df.groupby('super_category')['return_rate_percent'].mean().idxmax()}

🏷️ DISCOUNTS:
   • Products on discount: {len(df[df['discount_percent'] > 0])} ({len(df[df['discount_percent'] > 0])/len(df)*100:.1f}%)
   • Average discount offered: {df['discount_percent'].mean():.2f}%

🏆 TOP PERFORMING BRANDS:
   • By product count: {df['brand'].value_counts().idxmax()} ({df['brand'].value_counts().max()} products)
   • By sales volume: {df.groupby('brand')['monthly_sales_volume'].mean().idxmax()}
   • By rating: {df.groupby('brand')['avg_rating'].mean().idxmax()}
"""

print(insights)

# ============================================================================
# 16. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("16. GENERATING VISUALIZATIONS")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Recomart Product Catalog - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Super Category Distribution
ax1 = axes[0, 0]
super_cat_counts = df['super_category'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(super_cat_counts)))
ax1.pie(super_cat_counts, labels=super_cat_counts.index, autopct='%1.1f%%', colors=colors)
ax1.set_title('Products by Super Category', fontweight='bold')

# 2. Price Distribution
ax2 = axes[0, 1]
ax2.hist(df['base_price'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(df['base_price'].mean(), color='red', linestyle='--', label=f'Mean: ₹{df["base_price"].mean():.0f}')
ax2.set_xlabel('Base Price (₹)')
ax2.set_ylabel('Frequency')
ax2.set_title('Price Distribution', fontweight='bold')
ax2.legend()

# 3. Sales Distribution
ax3 = axes[0, 2]
ax3.hist(df['monthly_sales_volume'], bins=30, color='seagreen', edgecolor='white', alpha=0.7)
ax3.axvline(df['monthly_sales_volume'].mean(), color='red', linestyle='--', label=f'Mean: {df["monthly_sales_volume"].mean():.0f}')
ax3.set_xlabel('Monthly Sales Volume')
ax3.set_ylabel('Frequency')
ax3.set_title('Sales Distribution', fontweight='bold')
ax3.legend()

# 4. Average Price by Category
ax4 = axes[1, 0]
avg_price_by_cat = df.groupby('super_category')['base_price'].mean().sort_values(ascending=True)
bars = ax4.barh(avg_price_by_cat.index, avg_price_by_cat.values, color=plt.cm.viridis(np.linspace(0, 1, len(avg_price_by_cat))))
ax4.set_xlabel('Average Price (₹)')
ax4.set_title('Average Price by Category', fontweight='bold')
for i, v in enumerate(avg_price_by_cat.values):
    ax4.text(v + 5, i, f'₹{v:.0f}', va='center', fontsize=9)

# 5. Average Sales by Category
ax5 = axes[1, 1]
avg_sales_by_cat = df.groupby('super_category')['monthly_sales_volume'].mean().sort_values(ascending=True)
ax5.barh(avg_sales_by_cat.index, avg_sales_by_cat.values, color=plt.cm.plasma(np.linspace(0, 1, len(avg_sales_by_cat))))
ax5.set_xlabel('Average Monthly Sales')
ax5.set_title('Average Sales by Category', fontweight='bold')
for i, v in enumerate(avg_sales_by_cat.values):
    ax5.text(v + 20, i, f'{v:.0f}', va='center', fontsize=9)

# 6. Rating Distribution
ax6 = axes[1, 2]
ax6.hist(df['avg_rating'], bins=20, color='coral', edgecolor='white', alpha=0.7)
ax6.axvline(df['avg_rating'].mean(), color='blue', linestyle='--', label=f'Mean: {df["avg_rating"].mean():.2f}')
ax6.set_xlabel('Average Rating')
ax6.set_ylabel('Frequency')
ax6.set_title('Rating Distribution', fontweight='bold')
ax6.legend()

# 7. Profit Margin Distribution
ax7 = axes[2, 0]
ax7.hist(df['profit_margin_percent'], bins=25, color='mediumpurple', edgecolor='white', alpha=0.7)
ax7.axvline(df['profit_margin_percent'].mean(), color='red', linestyle='--', label=f'Mean: {df["profit_margin_percent"].mean():.1f}%')
ax7.set_xlabel('Profit Margin (%)')
ax7.set_ylabel('Frequency')
ax7.set_title('Profit Margin Distribution', fontweight='bold')
ax7.legend()

# 8. Return Rate vs Profit Margin
ax8 = axes[2, 1]
scatter = ax8.scatter(df['return_rate_percent'], df['profit_margin_percent'], 
                      c=df['avg_rating'], cmap='RdYlGn', alpha=0.6, s=50)
ax8.set_xlabel('Return Rate (%)')
ax8.set_ylabel('Profit Margin (%)')
ax8.set_title('Return Rate vs Profit Margin', fontweight='bold')
plt.colorbar(scatter, ax=ax8, label='Avg Rating')

# 9. Correlation Heatmap
ax9 = axes[2, 2]
corr_matrix = df[correlation_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax9, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax9.set_title('Correlation Matrix', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/purushottampandey/Documents/DM4ML/EDA_Visualizations.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved as 'EDA_Visualizations.png'")

# Additional visualizations
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Additional EDA Insights', fontsize=14, fontweight='bold')

# Top 10 Brands by Sales
ax10 = axes2[0, 0]
top_brands_sales = df.groupby('brand')['monthly_sales_volume'].mean().nlargest(10).sort_values()
ax10.barh(top_brands_sales.index, top_brands_sales.values, color='teal')
ax10.set_xlabel('Average Monthly Sales')
ax10.set_title('Top 10 Brands by Sales', fontweight='bold')

# Discount vs Sales
ax11 = axes2[0, 1]
ax11.scatter(df['discount_percent'], df['monthly_sales_volume'], alpha=0.5, c='orange')
ax11.set_xlabel('Discount (%)')
ax11.set_ylabel('Monthly Sales Volume')
ax11.set_title('Discount vs Sales', fontweight='bold')

# Price vs Rating
ax12 = axes2[1, 0]
ax12.scatter(df['base_price'], df['avg_rating'], alpha=0.5, c='purple')
ax12.set_xlabel('Base Price (₹)')
ax12.set_ylabel('Average Rating')
ax12.set_title('Price vs Rating', fontweight='bold')

# Products launched per year
ax13 = axes2[1, 1]
yearly_launches = df['launch_year'].value_counts().sort_index()
ax13.bar(yearly_launches.index.astype(str), yearly_launches.values, color='crimson', alpha=0.7)
ax13.set_xlabel('Launch Year')
ax13.set_ylabel('Number of Products')
ax13.set_title('Product Launches by Year', fontweight='bold')
ax13.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/purushottampandey/Documents/DM4ML/EDA_Additional_Insights.png', dpi=150, bbox_inches='tight')
print("✅ Additional insights saved as 'EDA_Additional_Insights.png'")

print("\n" + "=" * 80)
print("EDA COMPLETE! ✅")
print("=" * 80)
print("\nOutput files generated:")
print("  1. EDA_Visualizations.png - Main visualization dashboard")
print("  2. EDA_Additional_Insights.png - Additional analysis charts")

