import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("AB_NYC_2019.csv")
df = df[df.price != 0]

room_count = pd.crosstab(df.neighbourhood_group, df.room_type)
print(room_count)
room_count.to_csv('room_count.csv')

#Countplot of room types for each location
plt.figure(figsize=(10,10))
plot1 = sns.countplot(df['room_type'],hue=df['neighbourhood_group'], palette='plasma')
plt.title("Count of room types for each neighbourhood area")
plt.show(plot1)
FigMap1 = plot1.get_figure()
FigMap1.savefig("Countplot.png")

#Scatterplot of price distribution for each location
plt.figure(figsize=(10,10))
plot2 = sns.catplot(x="neighbourhood_group", y="price", kind="boxen", data=df);
plt.title("Scatterplot showing distribution of prices for each neighbourhood area")
plt.show(plot2)
plot2.savefig("Catplot.png")

df1 = df[df.price < 1000]
#Boxplot of prices less than 1000 for each location
plt.figure(figsize=(10,10))
plot3 = sns.boxplot(data=df1, x='neighbourhood_group',y='price',palette='plasma')
plt.title("Box plot of prices below 1000 for each neighbourhood area")
plt.show(plot3)
FigMap3 = plot3.get_figure()
FigMap3.savefig("Boxplot_1.png")

df2 = df[df.price > 1000]
#Boxplot of prices less than 1000 for each location
plt.figure(figsize=(10,10))
plot4 = sns.boxplot(data=df2, x='neighbourhood_group',y='price',palette='plasma')
plt.title("Box plot of prices above 1000 for each neighbourhood area")
plt.show(plot4)
FigMap4 = plot4.get_figure()
FigMap4.savefig("Boxplot_2.png")

#Heatmap showing mean prices of room types in each neighbourhood area
plt.figure(figsize=(12,12))
plot5 = sns.heatmap(df.groupby([
        'neighbourhood_group', 'room_type']).price.mean().unstack(),annot=True, fmt=".0f")
plt.title("Heat map showing mean prices of room types in each neighbourhood area")
plt.show(plot5)
FigMap5 = plot5.get_figure()
FigMap5.savefig("Heatmap.png")




