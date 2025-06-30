import numpy as np
import streamlit as st
import pandas as pd
import hmac
import cryptpandas as crp

PRES_SIMILARITIES_MATRIX_PATH = 'pres_similarities_matrix.parquet'
SESSION_SIMILARITIES_MATRIX_PATH = 'session_similarities_matrix.parquet'
NO_ABSTRACT_PRES_DATA_PATH = 'df_no_abstract.parquet'
ENCRYPTED_PRES_DATA_PATH = 'encrypted_df.crypt'
SESSION_DATA_PATH = 'df_sessions.parquet'

st.set_page_config(
    page_title="AIM 2025 Presentation Similarity",
    page_icon=":material/category_search:",
    layout="wide",
)

st.title("Presentation Similarity Exploration Tool")

# Password Check to unlock abstracts.
def password_entered():
    """Checks whether a password entered by the user is correct."""
    if hmac.compare_digest(st.session_state["password"], st.secrets["access_password"]):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the password.
    else:
        st.session_state["password_correct"] = False

@st.cache_data
def load_presentation_similarities():
    return pd.read_parquet(PRES_SIMILARITIES_MATRIX_PATH)

@st.cache_data  
def load_session_similarities():
    return pd.read_parquet(SESSION_SIMILARITIES_MATRIX_PATH)

@st.cache_data
def load_session_data():
    """Load session data with caching for speed."""
    return pd.read_parquet(SESSION_DATA_PATH)

@st.cache_data
def load_presentation_data(use_encrypted=False):
    """Load presentation data with caching for speed."""
    if use_encrypted:
        return crp.read_encrypted(
            path=ENCRYPTED_PRES_DATA_PATH, 
            password=st.secrets["df_password"]
        )
    else:
        return pd.read_parquet(NO_ABSTRACT_PRES_DATA_PATH)

# Show input for password.
st.text_input(
    "Password to view abstracts",
    type="password",
    on_change=password_entered,
    key="password",
)

# Load Base Presentation DataFrames
# Returns True if the password is validated.
if st.session_state.get("password_correct", False):
    df_presentations = load_presentation_data(use_encrypted=True)
else:
    df_presentations = load_presentation_data(use_encrypted=False)

#Load Session DataFrame
df_sessions = load_session_data()

# Load Similarity DataFrames
df_similarity  = load_presentation_similarities()
df_session_similarity = load_session_similarities()

with st.expander("Similarity Metric Descriptions"):
    st.markdown(
        '*All Similarity Metrics range from 0.0 (no relation) to 1.0 (identical). Scales are relevative and not absolute. They vary by model and cannot be compared across models. (e.g. A 0.6 is a "poor" similiarity with the nomic-embed-text-v1.5 model, but an "average" similiary with the cde-small-v1 model.*'
    )
    st.markdown(
        "**presentation_session_fit:**  This *presentation metric* is the average cosine similarity between a presentation and all others in its assigned session. It measures how similar a presentation is to others in its session. It does not include a presentation's similarity with itself, which is always 1.0."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(
        r"$PSS(p_i) = \frac{1}{|s_j| - 1} \sum_{\substack{p_k \in s_j \\ p_k \neq p_i}} sim(p_i, p_k)$"
    )
    st.write("where:")
    st.markdown(
        r"- $PSS(p_i)$ is the presentation_session_fit for presentation $p_i$,"
    )
    st.markdown(r"- $s_j$ is the session where presentation $p_i$ is assigned,")
    st.markdown(r"- $|s_j|$ is the number of presentations in session $s_j$,")
    st.markdown(r"- $p_k$ is a presentation in session $s_j$ other than $p_i$, and")
    st.markdown(
        r"- $sim(p_i, p_k)$ is the cosine similarity between presentation $p_i$ and presentation $p_k$."
    )
    st.write(
        "**session_coherence:** This *session metric* is the average cosine similarity between all presentations assigned to the same session. It is an overall indicator of how well a session focuses on one topic."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(
        r"$SS(s_j) = \frac{1}{|s_j|(|s_j| - 1)} \sum_{\substack{p_i \in s_j \\ p_k \in s_j \\ p_i \neq p_k}} sim(p_i, p_k)$"
    )
    st.write("where:")
    st.markdown(r"- $SS(s_j)$ is the session_coherence for session $s_j$,")
    st.markdown(
        r"- $\sum_{\substack{p_i \in s_j \\ p_k \in s_j \\ p_i \neq p_k}}$ is the sum the cosine similarities over all pairs of distinct presentations ($p_i$, $p_k$) within the session $s_j$, ignoring self-similarity, hence $p_i \neq p_k$, and"
    )
    st.markdown(
        r"- $|s_j|(|s_j| - 1)$ is number of all possible pairs of presentations $(p_i, p_k)$ within the session $s_j$."
    )
    st.write(
        "**Session Std Dev:** This *session metric* is the standard deviation of the presentation_session_fit scores of the presentations assigned to that session. Measures the variation in Presentation-Session scores.  session_coherence is a better measure of focus, but this metric can be used to identify sessions with outlier presentations."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(
        r"$SSD(s_j) = \sqrt{\frac{1}{|s_j| - 1} \sum_{p_i \in s_j} \left( PSS(p_i) - \overline{PSS(s_j)} \right)^2}$"
    )
    st.write("where:")
    st.markdown(
        r"- $SSD(s_j))$ is the Session Standard Deviation for session $s_j$, and"
    )
    st.markdown(
        r"- $\overline{PSS(s_j)}$ is the average presentation_session_fit for all presentations in session $s_j$."
    )
    st.write(
        "**Raw Deviation:** This *presentation metric* is the difference between a presentation's presentation_session_fit and its session's session_coherence. A direct measure of the difference in similarity of a presentation and its session."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(r"$RD(p_i) = PSS(p_i) - SS(s_j)$")
    st.write("where:")
    st.markdown(r"- $RD(p_i)$ is the Raw Deviation for presentation $p_i$.")
    st.write(
        "**Standardized Deviation:** This *presentation metric* is the Raw Deviation of the presentation divided by the Session Standard Deviation of the session to which it is assigned. This standardizes the similarity difference based on the variability in a session. This is analogous to a z-score. It is most useful for identifying single presentations that stand out from an otherwise very focused session."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(r"$SD(p_i) = \frac{RD(p_i)}{SSD(s_j)}$")
    st.write("where:")
    st.markdown(r"- $SD(p_i)$ is the Standardized Deviation for presentation $p_i$.")
    st.write(
        "**Session-Session Similarity:** This *session to session metric* is the average similarity between all presentations in one session with all those in another session. It indicates how similar the topic of one session is to the topic of another session."
    )
    st.write("It is calculated as:")
    # Using raw strings to perserve LaTex format.
    st.markdown(
        r"$SSS(s_j, s_m) = \frac{1}{|s_j| \cdot |s_m|} \sum_{p_i \in s_j} \sum_{p_k \in s_m} sim(p_i, p_k)$"
    )
    st.write("where:")
    st.markdown(
        r"- $SSS(s_j, s_m)$ is the Session-Session Similarity between session $s_j$ and session $s_m$,"
    )
    st.markdown(r"- $|s_j|$ is the number of presentations in session $s_j$,")
    st.markdown(r"- $|s_m|$ is the number of presentations in session $s_m$, and")
    st.markdown(
        r"- $\sum_{p_i \in s_j} \sum_{p_k \in s_m}$ is the sum of the cosine similarities over all pairs of presentations, where $p_i$ belongs to session $s_j$ and $p_k$ belongs to session $s_m$."
    )

tab_pres, tab_session = st.tabs(
    ["View Presentations", "View Sessions"]
)

with tab_pres:
    st.header("Presentations")
    with st.expander("**Instructions** Click to expand"):
        st.write(
            "Select a presentation by clicking on the checkbox. You can sort the presentation list or search as well."
        )
        st.write(
            "Once a presentation is selected, its abstract and the ten most similar presentations will appear in a list below."
        )
        st.write(
            "If you move your mouse over the table, a menu will appear in the top left corner that lets you search within the table or download. Clicking on columns will let you sort by the column too."
        )
        st.write(
            "If text is cut off, click twice on an cell to see the full text. You can scroll left-right and up-down in the table."
        )
        st.write("Similarity scores range from 0.0 (not similar) to 1.0 (identical).")
        st.write(
            "The leftmost column is a checkbox column. Click to select a presentation. This may blend with the background on dark themes."
        )

    event = st.dataframe(
        df_presentations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Abstract ID": st.column_config.NumberColumn(format="%i"),
            "presentation_session_fit": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Session Std Dev": None,
            "Raw Deviation": st.column_config.NumberColumn(format="%.3f"),
            "Standardized Deviation": st.column_config.NumberColumn(format="%.3f"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:  # Check if a presentation has been selected.
        st.header("Selected Presentation:")
        selected_pres = df_presentations.iloc[
            event.selection.rows
        ]  # Create a dataframe from the selected presentation row.
        st.write(
            selected_pres.iloc[0]["Title"]
        )  # It is necessary to request the first row, [0], since it is a dataframe and not just one entry.
        st.header("Most Similar Presentations")
        similar_presentations = df_similarity.loc[
            selected_pres.iloc[0].name
        ].sort_values(
            ascending=False
        )  # Create a Series with the  most similar presentations
        # Remove the selected presentation itself from the similar presentations
        similar_presentations = similar_presentations.drop(selected_pres.iloc[0].name)
        # Build the similarity dataframe. Add the similarity score and similarity rank to the dataframe and show it.
        similar_df = df_presentations.loc[similar_presentations.index]
        similar_df.insert(0, "Similarity Score", similar_presentations)
        similar_df.insert(0, "Similarity Rank", np.arange(1, similar_df.shape[0] + 1))
        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Abstract ID": st.column_config.NumberColumn(format="%i"),
                "presentation_session_fit": None,
                "Session Std Dev": None,
                "Raw Deviation": None,
                "Standardized Deviation": None,
            },
        )
with tab_session:
    st.header("Sessions")
    cluster_session_sizes_df = df_sessions["session_size"]
    cluster_session_sizes_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sizes_df = cluster_session_sizes_df.rename("Presentations Count")
    cluster_session_sizes_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Size Distribution") 
    st.bar_chart(cluster_session_sizes_df, x_label="Session Name", y_label="Presentations Count") 

    cluster_session_sim_df = df_sessions["session_coherence"]
    cluster_session_sim_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sim_df = cluster_session_sim_df.rename("Session Coherence Score")
    cluster_session_sim_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Coherence Distribution") 
    st.bar_chart(cluster_session_sim_df, x_label="Session Name", y_label="Session Coherence Score")


    with st.expander("**Instructions** Click to expand"):
        st.write(
            "Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table."
        )
    event_session = st.dataframe(
        df_sessions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "session_coherence": st.column_config.NumberColumn(format="%.3f"),
            "Session Std Dev": st.column_config.NumberColumn(format="%.3f"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    if event_session.selection.rows:  # Check if a session has been selected.
        st.header("Session Details")
        selected_session_df = df_sessions.iloc[
            event_session.selection.rows
        ]  # Create a dataframe from the selected session row.
        selected_session = selected_session_df.iloc[0]["cluster_id"]
        st.subheader(selected_session)
        st.write(
            f"**session_coherence:** {selected_session_df.iloc[0]['session_coherence']:.3f}"
        )
        df_selected_session = df_presentations[
            df_presentations["Session Code"] == selected_session
        ]
        if (
            "Abstract" in df_selected_session
        ):  # Check if the dataframe has the abstract in it to determine how to display.
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=[
                    "presentation_session_fit",
                    "Standardized Deviation",
                    "Abstract ID",
                    "Title",
                    "Abstract",
                ],
                column_config={
                    "Abstract ID": st.column_config.NumberColumn(format="%i"),
                    "presentation_session_fit": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Session Std Dev": None,
                    "Raw Deviation": st.column_config.NumberColumn(format="%.3f"),
                    "Standardized Deviation": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
            )
        else:
            st.dataframe(
                df_selected_session,
                use_container_width=True,
                hide_index=True,
                column_order=[
                    "presentation_session_fit",
                    "Standardized Deviation",
                    "Abstract ID",
                    "Title",
                ],
                column_config={
                    "Abstract ID": st.column_config.NumberColumn(format="%i"),
                    "presentation_session_fit": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    "Session Std Dev": None,
                    "Raw Deviation": st.column_config.NumberColumn(format="%.3f"),
                    "Standardized Deviation": st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                },
            )
        st.header("Most Similar Sessions")
        # Create a Series with the  most similar sessions
        similar_sessions = df_session_similarity[selected_session].sort_values(
            ascending=False
        )
        # Remove the selected presentation itself from the similar presentations
        similar_sessions = similar_sessions.drop(selected_session)
        similar_sessions_df = pd.DataFrame(similar_sessions)
        st.write("Other sessions that are most similar to:")
        st.subheader(similar_sessions_df.columns[0])
        st.write("This list is initially sorted by similarity to the selected session.")
        similar_sessions_df = similar_sessions_df.rename(
            columns={
                similar_sessions_df.columns[0]: "Session-Session Similarity Score",
            }
        )
        similar_sessions_df.insert(
            0, "Session Similarity Rank", np.arange(1, similar_sessions_df.shape[0] + 1)
        )
        st.dataframe(
            similar_sessions_df,
            use_container_width=True,
            hide_index=False,
        )
