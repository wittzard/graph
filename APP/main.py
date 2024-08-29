import streamlit as st

pg = st.navigation([st.Page("Graph_test.py"),
                    st.Page("plot_3d_rotation_animation.py"),
                    st.Page("Design_Layout_Analysis.py"),
                    st.Page("Graph_animation.py"),
                    st.Page("Layout_simulation.py")])
pg.run()