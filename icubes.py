#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("ecommerce_sales.csv")


# In[4]:


df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y', errors='coerce')


# In[5]:


df = df.dropna(subset=['order_date'])


# In[6]:


df['computed_total_price'] = df['quantity'] * df['unit_price']
df['total_price'] = df['total_price'].combine_first(df['computed_total_price'])


# In[7]:


df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['day_of_week'] = df['order_date'].dt.day_name()


# In[8]:


monthly_sales = df.groupby(df['order_date'].dt.to_period("M"))['total_price'].sum().reset_index()
monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()


# In[9]:


top_categories = df.groupby('product_category')['total_price'].sum().sort_values(ascending=False).head(10)


# In[10]:


quantity_per_category = df.groupby('product_category')['quantity'].sum().sort_values(ascending=False)


# In[11]:


ig, axes = plt.subplots(3, 1, figsize=(12, 18))

sns.lineplot(data=monthly_sales, x='order_date', y='total_price', ax=axes[0], marker='o')
axes[0].set_title('📈 Monthly Sales Trend')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Revenue')


# In[12]:


sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[1], palette='viridis')
axes[1].set_title('🏆 Top 10 Product Categories by Revenue')
axes[1].set_xlabel('Revenue')


# In[13]:


sns.barplot(x=quantity_per_category.values, y=quantity_per_category.index, ax=axes[2], palette='magma')
axes[2].set_title('📦 Quantity Sold per Product Category')
axes[2].set_xlabel('Quantity Sold')


# In[14]:


plt.tight_layout()
plt.show()


# In[15]:


plt.show()


# In[16]:


# Unique customers
num_customers = df['customer_id'].nunique()

# Orders per customer
orders_per_customer = df.groupby('customer_id')['order_id'].nunique()

# Revenue per customer
revenue_per_customer = df.groupby('customer_id')['total_price'].sum()

# Average order value per customer
aov_per_customer = revenue_per_customer / orders_per_customer

# Recency: last order date
last_order = df.groupby('customer_id')['order_date'].max()
recency_days = (df['order_date'].max() - last_order).dt.days

# RFM Table
rfm = pd.DataFrame({
    'Recency': recency_days,
    'Frequency': orders_per_customer,
    'Monetary': revenue_per_customer
})

# Scoring (1–4)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)

# Print RFM Head
print("Top Customers by RFM Score:")
print(rfm.sort_values("RFM_Score", ascending=False).head())



# Revenue by Region
revenue_by_region = df.groupby('region')['total_price'].sum().sort_values(ascending=False)
orders_by_region = df.groupby('region')['order_id'].nunique().sort_values(ascending=False)
aov_by_region = revenue_by_region / orders_by_region

# Plot Regional Metrics
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

sns.barplot(x=revenue_by_region.values, y=revenue_by_region.index, ax=axes[0], palette="Blues_d")
axes[0].set_title("🌍 Total Revenue by Region")
axes[0].set_xlabel("Revenue")

sns.barplot(x=orders_by_region.values, y=orders_by_region.index, ax=axes[1], palette="Greens_d")
axes[1].set_title("📦 Number of Orders by Region")
axes[1].set_xlabel("Orders")

sns.barplot(x=aov_by_region.values, y=aov_by_region.index, ax=axes[2], palette="Oranges_d")
axes[2].set_title("💰 Average Order Value by Region")
axes[2].set_xlabel("AOV")

plt.tight_layout()
plt.show()

# Region Summary Table
region_summary = pd.DataFrame({
    'Total Revenue': revenue_by_region,
    'Number of Orders': orders_by_region,
    'Average Order Value': aov_by_region.round(2)
})
print(region_summary.reset_index())


 #Payment Method Breakdown


# Revenue by Payment Method
revenue_by_payment = df.groupby('payment_method')['total_price'].sum().sort_values(ascending=False)
orders_by_payment = df.groupby('payment_method')['order_id'].nunique().sort_values(ascending=False)
aov_by_payment = revenue_by_payment / orders_by_payment

# Plot Payment Breakdown
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

sns.barplot(x=revenue_by_payment.values, y=revenue_by_payment.index, ax=axes[0], palette="Purples_d")
axes[0].set_title(" Total Revenue by Payment Method")

sns.barplot(x=orders_by_payment.values, y=orders_by_payment.index, ax=axes[1], palette="BuGn_d")
axes[1].set_title(" Number of Orders by Payment Method")

sns.barplot(x=aov_by_payment.values, y=aov_by_payment.index, ax=axes[2], palette="YlOrBr")
axes[2].set_title(" AOV by Payment Method")

plt.tight_layout()
plt.show()

# Summary Table
payment_summary = pd.DataFrame({
    'Total Revenue': revenue_by_payment,
    'Number of Orders': orders_by_payment,
    'Average Order Value': aov_by_payment.round(2)
})
print(payment_summary.reset_index())


# In[ ]:




