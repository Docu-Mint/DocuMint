import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# Read the content of the file to extract the data
file_path = './fine_tune_logs.txt'
data = []
with open(file_path, 'r') as file:
    for line in file:
        # Remove newline characters and outer curly braces
        line = line.strip().replace('{', '').replace('}', '')
        # Split the line into key-value pairs
        key_value_pairs = line.split(', ')
        # Create a dictionary for each line
        data_dict = {item.split(': ')[0]: float(item.split(': ')[1]) for item in key_value_pairs}
        data.append(data_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
df.head()

# Correcting the column names which are incorrectly quoted
df.columns = [col.strip("'") for col in df.columns]

fs = 20
# Plotting the loss data again
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(df['epoch'], df['loss'], label='Loss', color='orange', linewidth=2)
plt.xlabel('Epoch', fontsize=fs)  # Increased font size
plt.ylabel('Fine-tuning Loss', fontsize=fs)  # Increased font size
#plt.title('Training Loss per Epoch', fontsize=fs)  # Title with default size as per request
plt.grid(True, linestyle='--')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(True)
plt.legend(fontsize=fs-2)
plt.tight_layout()
plt.savefig('./fine_tune_loss_plot.png')
# plt.show()
