import streamlit as st
import pandas as pd
import numpy as np
import folium
import time 


st.title("My frist streamlit web")
st.write("Created by wittzard")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("C://Users/plunm/Test/Model/Data/usa.csv")

st.write(df)
st.map(df)

def create_map(map_dataframe):
    map_center = [df['lat'].mean(), df['lon'].mean()]
    map = folium.Map(location=map_center, zoom_start=4, tiles="cartodb positron")
    # At all customer node
    for _,row in map_dataframe.iterrows():
        folium.Circle(
            location = [row['lat'],row['lon']],
            radius = 2,
            color = 'red',
            fill_color='red',
            fill_opacity=0.6,
            tooltip=row['name'],
        ).add_to(map)
    # At depot
    folium.Circle(
        location=[df.iloc[0]['lat'],df.iloc[0]['lon']],
        radius=2,  # Radius in pixels
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        tooltip=df.iloc[0]['name']
    ).add_to(map)

    coordinates = []
    n = len(df)
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            coordinate = []
            point1 = [df.iloc[i]['lat'], df.iloc[i]['lon']]
            point2 = [df.iloc[j]['lat'], df.iloc[j]['lon']]
            coordinate.append(point1)
            coordinate.append(point2)
            coordinates.append(coordinate)
    
    # Add Edge
    for i in range(len(coordinates)):
        folium.PolyLine(
            locations=coordinates[i],
            color='red',
            weight=1,
            tooltip="TSP"
        ).add_to(map)

create_map(df)

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")
st.button("Run it again2")

import streamlit as st
import pandas as pd
import numpy as np

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)