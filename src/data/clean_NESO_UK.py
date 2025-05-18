import pandas as pd 

df1 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2011-2016.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')
df2 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2017.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')
df3 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2018_4.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')

# Concatenation of the three datasets
df = pd.concat([df1,df2,df3], ignore_index=True)


print(df.isnull().sum().sort_values(ascending=False))

# Drop column I014_TSD.1 (duplicate of I014_ND)
df.drop(columns=['I014_TSD.1'], inplace=True)

# Reconstruct timestamps by adding 30-minute intervals to the settlement date
df['datetime'] = df['SETTLEMENT_DATE'] + pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='min')

# Set this datetime column as the index for time series analysis
df.set_index('datetime', inplace=True)

# Export the cleaned and indexed DataFrame to a CSV file
df.to_csv('D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/cleaned_DemandData_2011-2018.csv')