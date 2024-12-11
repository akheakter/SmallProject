import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from datetime import datetime

# (A) Load and prepare the dataset
df = pd.read_csv('airline1.csv')
print("\n Dataset Overview")
print(df.head())

# Convert 'Date' column to datetime format and extract date components
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Display initial insights
print("\n Dataset conert into day month year:")
print(df.head())

print("\nDataset Summary Information:")
print(df.info())

print("\nMissing Values Summary:")
print(df.isnull().sum())

print("\nNumber of Unique Values per Column:")
print(df.nunique())

print("\nDuplicate Rows Count:", df.duplicated().sum())

print("\nStatistical Summary of Dataset:")
print(df.describe())


# Yearly passenger records
print("\nPassenger Counts by Year:")
df['Year'].value_counts()


def plot_histogram(column, color, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=25, alpha=0.7, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

plot_histogram('Number', 'blue', 'Distribution of Passenger Numbers', 'Passenger Count (Thousands)', 'Frequency')
plot_histogram('Revenue', 'green', 'Distribution of Revenue', 'Revenue (Thousands of Euros)', 'Frequency')

# (C) Aggregated insights based on Year and Month
print("\nAverage Passengers by Year:")
print(df.groupby('Year')['Number'].mean())

print("\nAverage Passengers by Month:")
print(df.groupby('Month')['Number'].mean())


monthly_avg = df.groupby(['Year', 'Month'])['Number'].mean().unstack()
monthly_avg.plot(kind='bar', figsize=(12, 6), legend=True)
plt.title("Monthly Average Passenger Numbers by Year")
plt.xlabel("Year")
plt.ylabel("Average Passenger Numbers (Thousands)")
plt.grid()
plt.show()



# Extract the passenger numbers from the dataset
passenger_numbers = df['Number'].to_numpy()

# Calculate the length of the dataset
n = len(passenger_numbers)

# Perform the Fourier Transform
fft_transform = fft(passenger_numbers)

# Compute the corresponding frequencies
frequencies = fftfreq(n, d=1)  # Assumes daily intervals, so d=1

# Calculate the Power Spectrum (magnitude squared of the Fourier coefficients)
power_spectrum = np.abs(fft_transform) ** 2

# Filter only positive frequencies (ignore negative frequencies as they are redundant)
positive_freq_indices = frequencies > 0
positive_frequencies = frequencies[positive_freq_indices]
positive_power_spectrum = power_spectrum[positive_freq_indices]

# Print results
print("Fourier Transform Coefficients:")
print(fft_transform[:10])

print("\nFrequencies (1/Days):")
print(frequencies[:10])
          
print("\nPower Spectrum:")
print(power_spectrum[:10])

print("\nPositive Frequencies:")
print(positive_frequencies[:10])

print("\nPositive Power Spectrum:")
print(positive_power_spectrum[:10])




# Group data by month and calculate the average daily passengers
monthly_avg_passengers = df.groupby('Month')['Number'].mean()

# Define month names for better readability on the x-axis
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(month_names, monthly_avg_passengers, color='orange', alpha=0.8, label='Average Daily Passengers')

# Add labels, title, legend, and student ID
plt.xlabel('Month')
plt.ylabel('Passengers (thousands)')
plt.title('Average Daily Passengers by Month')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(0.2, 0.95, 'Student ID: 23029819', transform=plt.gca().transAxes, color='green', fontsize=12, fontweight='bold')

# Show the plot
plt.show()



# Calculate monthly averages
monthly_avg = df.groupby(['Year', 'Month'])['Number'].mean().reset_index()

# Prepare data for plotting
monthly_avg['Month_Year'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(str).str.zfill(2)
monthly_avg['Month_Year_Label'] = monthly_avg['Month'].apply(lambda x: datetime(1900, x, 1).strftime('%B'))

# Fourier series approximation (first 8 terms)
n_terms = 8
approximation = np.real(np.fft.ifft(fft_transform[:n_terms]))

# Generate Figure 1
plt.figure(figsize=(12, 6))

# Bar chart of monthly averages
plt.bar(
    monthly_avg['Month_Year_Label'],
    monthly_avg['Number'],
    alpha=0.6,
    label='Monthly Average Passengers',
    color='orange'
)

# Overlay Fourier series approximation
plt.plot(
    range(len(approximation)),
    approximation[:len(monthly_avg)],
    label='Fourier Approximation (8 terms)',
    linestyle='--'
)

# Add labels and title
plt.xlabel('Month')
plt.ylabel('Average Number of Passengers')
plt.title('Monthly Average Daily Passenger Numbers (Fourier Approximation Included)')
plt.xticks(rotation=45)
plt.legend()

# Show the plot
plt.tight_layout()
plt.text(0.2, 0.95, 'Student ID: 23029819', transform=plt.gca().transAxes, color='green', fontsize=12, fontweight='bold')

plt.show()


# Handle divide-by-zero for period calculations
periods = np.zeros_like(frequencies)  # Initialize periods array
nonzero_indices = frequencies != 0   # Identify nonzero frequencies
periods[nonzero_indices] = 1 / frequencies[nonzero_indices]  # Calculate periods for nonzero frequencies

# Filter periods between 7 days (1 week) and 365 days (1 year)
period_range = (7, 365)
valid_period_indices = (frequencies > 0) & (periods >= period_range[0]) & (periods <= period_range[1])

# Filtered periods and power
filtered_periods = periods[valid_period_indices]
filtered_power = power_spectrum[valid_period_indices]

# Identify periods and corresponding power within the range
filtered_periods = periods[valid_period_indices]
filtered_power = power_spectrum[valid_period_indices]

# Find the periods corresponding to the highest and second-highest power
sorted_indices = np.argsort(filtered_power)[::-1]  # Indices sorted by power, descending
highest_period = filtered_periods[sorted_indices[0]]
second_highest_period = filtered_periods[sorted_indices[1]]

# Create the refined power spectrum plot focusing on 7 to 365 days range
plt.figure(figsize=(12, 6))
plt.bar(filtered_periods, filtered_power, width=10, alpha=0.7, color='orange',label='Power Spectrum (7-365 days)')

# Add labels, title, and legend
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title(f'Power Spectrum (Periods: 7 to 365 Days)')
plt.legend()

# Show the plot
plt.tight_layout()
plt.text(0.2, 0.95, 'Student ID: 23029819', transform=plt.gca().transAxes, color='green', fontsize=12, fontweight='bold')
plt.show()


# Filter periods between 7 days (1 week) and 365 days (1 year)
period_range = (7, 365)
valid_period_indices = (frequencies > 0) & (periods >= period_range[0]) & (periods <= period_range[1])
# Get the total revenue for 2021 and 2022
revenue_2021 = df[df['Year'] == 2021]['Revenue'].sum()
revenue_2022 = df[df['Year'] == 2022]['Revenue'].sum()

print(f"Value X : {revenue_2021:.2f}")
print(f"Value Y : {revenue_2022:.2f}")

# Plot the filtered power spectrum
plt.figure(figsize=(12, 6))
plt.bar(periods[valid_period_indices], power_spectrum[valid_period_indices], width=10, alpha=0.7, color='orange',label='Power Spectrum (7-365 days)')

# Add labels, title, and legend
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.title(f'Power Spectrum (Periods: 7 to 365 Days)')
plt.legend()

# Annotate with student ID
plt.text(x=100, y=max(power_spectrum[valid_period_indices]) * 0.95,s=f'Total Revenue 2021 (X): {revenue_2021:.2f}\nTotal Revenue 2022 (Y): {revenue_2022:.2f}',fontsize=10,bbox=dict(facecolor='white', alpha=0.6))

# Show the plot
plt.tight_layout()

plt.text(0.7, 0.95, 'Student ID: 23029819', transform=plt.gca().transAxes, color='green', fontsize=12, fontweight='bold')

plt.show()
