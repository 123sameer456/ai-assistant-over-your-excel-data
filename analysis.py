
import pandas as pd
df = pd.read_csv("sampledata.csv")


# Total Sales (NetAmount) aur Average Sale
# print("Total Sales:", df["NetAmount"].sum())
# print("Average Sale:", df["NetAmount"].mean())


# Sales by Branch
# branch_sales = df.groupby("Branch")["NetAmount"].sum().reset_index()
# print(branch_sales)


# Sales by Category
# category_sales = df.groupby("TopLevelCategory")["NetAmount"].sum().reset_index()
# print(category_sales)

# Top 10 Products by Sales
# top_products = df.groupby("ProductName")["NetAmount"].sum().reset_index().sort_values(by="NetAmount", ascending=False).head(10)
# print(top_products)

# Branch-wise Quantity and Sales
# branch_summary = df.groupby("Branch").agg({
#     "Quantity": "sum",
#     "NetAmount": "sum",
#     "InvoiceDiscount": "sum",
#     "GST": "sum"
# }).reset_index()
# print(branch_summary)


# Channel-wise Sales (WalkIn, DineIn, etc.)
# channel_sales = df[[
#     "ST_WalkInAmt", "ST_HomeDeliveryAmt", "ST_DineInAmt", 
#     "ST_TakeAwayAmt", "ST_DineOutAmt", "ST_FoodPandaAmt"
# ]].sum()
# print(channel_sales)


# Monthly Sales Trend



# Convert InvoiceDate to proper datetime
# df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", infer_datetime_format=True)

# print(df["InvoiceDate"].head(20))


#                  NaT
# 1    2025-08-19 23:19:36
# 2                    NaT
# 3    2025-08-19 08:34:18
# 4    2025-08-19 19:21:18
