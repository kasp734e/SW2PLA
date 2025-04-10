import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read the CSV file
df = pd.read_csv('ElonTweets.csv')

# Print basic info about the dataset to understand its structure
print("First few rows:")
print(df.head())
print("\nColumns in dataset:", df.columns.tolist())

# Try to identify the date column (column names may vary)
date_columns = [col for col in df.columns if any(word in col.lower() 
                for word in ['date', 'time', 'created'])]

if date_columns:
    date_column = date_columns[0]
else:
    # If no obvious date column, assume it's the first column
    date_column = df.columns[0]
    
print(f"Using '{date_column}' as the date column")

# Convert the date column to datetime
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
df = df.dropna(subset=[date_column])  # Remove rows with invalid dates

# Check if the datetime objects have timezone info
has_tz = df[date_column].dt.tz is not None

# Filter data from January 1, 2012 onwards
if has_tz:
    # Get the timezone from the data
    tz_info = df[date_column].dt.tz
    cutoff_date = pd.to_datetime('2012-01-01').tz_localize(tz_info)
    print(f"Data has timezone: {tz_info}")
else:
    cutoff_date = pd.to_datetime('2012-01-01')

df = df[df[date_column] >= cutoff_date]
print(f"\nFiltered to data from {cutoff_date.strftime('%Y-%m-%d')} onwards")
print(f"Remaining tweets: {len(df)}")

# Sort data by date
df = df.sort_values(by=date_column)

# Convert dates to numeric format for regression (days since first tweet)
first_date = df[date_column].min()
numeric_days = np.array([(date - first_date).total_seconds() / (60*60*24) for date in df[date_column]])

# Create a daily count for visualization
df_daily = df.groupby(df[date_column].dt.date).size().reset_index(name='count')
df_daily = df_daily.rename(columns={df_daily.columns[0]: 'date'})  # Rename the first column to 'date'
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily = df_daily.sort_values('date')

# Perform exponential regression on daily counts
numeric_x = np.array([(date - df_daily['date'].min()).total_seconds() / (60*60*24) for date in df_daily['date']])
daily_counts = np.array(df_daily['count'])

# Filter out zeros for log transform
mask = daily_counts > 0
numeric_x_filtered = numeric_x[mask]
daily_counts_filtered = daily_counts[mask]

# Perform regression on log-transformed data (for exponential form y = a*e^(b*x))
log_y = np.log(daily_counts_filtered)
b, log_a = np.polyfit(numeric_x_filtered, log_y, 1)
a = np.exp(log_a)

# Generate exponential regression curve
exp_regression_line = a * np.exp(b * numeric_x)

# Calculate R-squared for exponential fit
log_regression_line = log_a + b * numeric_x_filtered
ss_total = np.sum((log_y - np.mean(log_y)) ** 2)
ss_residual = np.sum((log_y - log_regression_line) ** 2)
exp_r_squared = 1 - (ss_residual / ss_total)

# Add linear regression
linear_coef = np.polyfit(numeric_x, daily_counts, 1)
linear_regression_line = linear_coef[0] * numeric_x + linear_coef[1]

# Calculate R-squared for linear fit
ss_total_lin = np.sum((daily_counts - np.mean(daily_counts)) ** 2)
ss_residual_lin = np.sum((daily_counts - linear_regression_line) ** 2)
lin_r_squared = 1 - (ss_residual_lin / ss_total_lin)

# Add polynomial regression (degree 2)
poly2_coef = np.polyfit(numeric_x, daily_counts, 2)
poly2_regression_line = poly2_coef[0] * numeric_x**2 + poly2_coef[1] * numeric_x + poly2_coef[2]

# Calculate R-squared for polynomial (degree 2) fit
ss_residual_poly2 = np.sum((daily_counts - poly2_regression_line) ** 2)
poly2_r_squared = 1 - (ss_residual_poly2 / ss_total_lin)

# Add polynomial regression (degree 3)
poly3_coef = np.polyfit(numeric_x, daily_counts, 3)
poly3_regression_line = (poly3_coef[0] * numeric_x**3 + poly3_coef[1] * numeric_x**2 + 
                        poly3_coef[2] * numeric_x + poly3_coef[3])

# Calculate R-squared for polynomial (degree 3) fit
ss_residual_poly3 = np.sum((daily_counts - poly3_regression_line) ** 2)
poly3_r_squared = 1 - (ss_residual_poly3 / ss_total_lin)

# Add logarithmic regression (y = a + b*ln(x))
# Add 1 to numeric_x to avoid log(0)
x_for_log = numeric_x + 1
log_coef = np.polyfit(np.log(x_for_log), daily_counts, 1)
log_regression_line = log_coef[0] * np.log(x_for_log) + log_coef[1]

# Calculate R-squared for logarithmic fit
ss_residual_log = np.sum((daily_counts - (log_coef[0] * np.log(x_for_log) + log_coef[1])) ** 2)
log_r_squared = 1 - (ss_residual_log / ss_total_lin)

# Calculate prediction for January 1, 2030
target_date = pd.to_datetime('2030-01-01')
if has_tz:
    target_date = target_date.tz_localize(df[date_column].dt.tz)
    
target_days = (target_date - first_date).total_seconds() / (60*60*24)

# Calculate predictions for each model
exp_prediction = a * np.exp(b * target_days)
linear_prediction = linear_coef[0] * target_days + linear_coef[1]
poly2_prediction = poly2_coef[0] * target_days**2 + poly2_coef[1] * target_days + poly2_coef[2]
poly3_prediction = (poly3_coef[0] * target_days**3 + poly3_coef[1] * target_days**2 + 
                   poly3_coef[2] * target_days + poly3_coef[3])
log_prediction = log_coef[0] * np.log(target_days + 1) + log_coef[1]

# Print the predictions
print(f"\nPredictions for tweets on January 1, 2030:")
print(f"Exponential model: {exp_prediction:.1f} tweets")
print(f"Linear model: {linear_prediction:.1f} tweets")
print(f"Polynomial degree 2 model: {poly2_prediction:.1f} tweets")
print(f"Polynomial degree 3 model: {poly3_prediction:.1f} tweets")
print(f"Logarithmic model: {log_prediction:.1f} tweets")

# Create the plot
plt.figure(figsize=(15, 8))
plt.style.use('seaborn-v0_8-whitegrid')  # Using a cleaner style

# Plot tweets per day as bars
plt.bar(df_daily['date'], df_daily['count'], alpha=0.4, color='lightgray', label='Tweets per day')

# Plot all regression lines with different colors, styles and increased width
plt.plot(df_daily['date'], exp_regression_line, 'r-', linewidth=2.5, 
         label=f'Exponential (R² = {exp_r_squared:.2f})')
plt.plot(df_daily['date'], linear_regression_line, 'g--', linewidth=2.5, 
         label=f'Linear (R² = {lin_r_squared:.2f})')
plt.plot(df_daily['date'], poly2_regression_line, 'b-.', linewidth=2.5, 
         label=f'Polynomial degree 2 (R² = {poly2_r_squared:.2f})')
plt.plot(df_daily['date'], poly3_regression_line, 'c:', linewidth=2.5, 
         label=f'Polynomial degree 3 (R² = {poly3_r_squared:.2f})')
plt.plot(df_daily['date'], log_regression_line, 'm-', linewidth=2.5, 
         label=f'Logarithmic (R² = {log_r_squared:.2f})')

# Add labels and formatting
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Tweets', fontsize=14)
plt.title('Elon Musk Tweets per Day with Multiple Regression Models', fontsize=16, fontweight='bold')

# Format x-axis dates for better readability
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.yticks(fontsize=12)

# Position legend outside the plot for clarity
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12)

# Adjust layout to ensure everything fits
plt.tight_layout()

# Show the plot
plt.show()

# Display regression statistics
print(f"\nRegression Results:")
print(f"Exponential: Slope: {b:.4f}, Intercept: {a:.4f}, R-squared: {exp_r_squared:.4f}")
print(f"Linear: Coefficients: {linear_coef}, R-squared: {lin_r_squared:.4f}")
print(f"Polynomial degree 2: Coefficients: {poly2_coef}, R-squared: {poly2_r_squared:.4f}")
print(f"Polynomial degree 3: Coefficients: {poly3_coef}, R-squared: {poly3_r_squared:.4f}")
print(f"Logarithmic: Coefficients: {log_coef}, R-squared: {log_r_squared:.4f}")
