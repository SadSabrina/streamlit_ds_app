import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path: str):
    return pd.read_csv(path, index_col=0)

#pie chart function
def get_rare_pie(data: pd.Series, limits_count: int = 5):

    """Function to draw pie chart with rare categories"""
    limits_dict = {}

    for i in range(1, limits_count + 1):
        limits_dict[i] = int(len(data) / 100 * i + 1)

    rare_values = {}
    data_to_pie_dict_categories = {}

    for i in range(1, limits_count + 1):

        if i == 1:
            rare_values[i] = data.value_counts()[data.value_counts() < limits_dict[i]].index

        else:
            rare_values[i] = data.value_counts()[
                (data.value_counts() < limits_dict[i]) & (data.value_counts() >= limits_dict[i - 1])].index

        data_to_pie_dict_categories[
            f'Categories ({len(rare_values[i])}) in which there are less samples than {i}% of data'] = len(
            rare_values[i])

    pie_labels_categories = data_to_pie_dict_categories.keys()
    pie_values_categories = data_to_pie_dict_categories.values()

    fig = plt.figure(figsize=(6, 6))

    plt.pie(pie_values_categories, autopct='%1.0f%%',
            colors=sns.color_palette('Set2'), shadow=True,
            wedgeprops={"edgecolor": "white",
                        'linewidth': 2,
                        'antialiased': True})
    plt.legend(pie_labels_categories, bbox_to_anchor=(1, 0.7))
    plt.title('Categories in which samples are less than % of the data', weight='bold')
    plt.show()

    return fig