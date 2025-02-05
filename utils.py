import inspect
import textwrap
import streamlit as st

def show_code(func):
    """
    Displays the source code of a given function in the Streamlit sidebar.
    Check the box in the sidebar to view the code.
    """
    if st.sidebar.checkbox("Show code", value=False):
        st.markdown("## Code")
        source_lines, _ = inspect.getsourcelines(func)
        st.code(textwrap.dedent("".join(source_lines)))
