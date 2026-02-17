import pandas as pd

df = pd.read_csv('./data_madrid/dataset-copy.csv')


max_price = df['Actual price'].max()
min_price = df['Actual price'].min()

print(f"Most expensive house: {max_price:,.2f} €")
print(f"Cheapest house: {min_price:,.2f} €")


print("\nTop 5 most expensive houses:")
print(df['Actual price'].nlargest(5))


print("\nStatistical summary of the Price column:")
print(df['Actual price'].describe())
