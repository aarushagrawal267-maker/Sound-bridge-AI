import streamlit as st

from src import ui

engine = ui.get_engine()
ui.page_header("About", "About SoundBridge AI",
               "An independent research platform asking whether a recommender can balance personal taste, musical "
               "diversity and visibility for lesser-known artists — and explain itself while doing it.")

c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    st.markdown("""
### The problem
Listeners often find it hard to discover music outside what they already play, while lesser-known artists struggle for
visibility in popularity-driven environments. Recommenders trained to maximise immediate relevance tend to amplify both
effects.

### The approach
1. **Learn the sound.** The SoundBridge Encoder, a multi-task neural network trained from scratch on 79k real tracks,
   maps audio features into a 32-dimensional sound space (contrastive same-artist learning + genre + popularity heads).
2. **Profile the listener.** Seed songs become a Musical DNA: interpretable feature summaries plus embedding centroids.
3. **Rank with intent.** Seven strategies, from plain similarity to a hybrid objective that balances relevance, novelty,
   diversity, genre exploration and repetition — all configurable.
4. **Explain everything.** Feature comparisons, an exact cosine decomposition and integrated-gradients attribution.
5. **Measure, don't assert.** A reproducible Experiment Lab, real-feedback evaluation and a pilot-study workflow.
""")
with c2:
    st.markdown("### Research questions")
    rq = [
        ("How can audio features identify musical similarity?", "Recommendation Engine · held-out retrieval"),
        ("Can content-based filtering recommend relevant new songs?", "Experiment Lab · relevance proxies"),
        ("How does popularity influence the distribution of recommendations?", "Experiment Lab · popularity-ranked baseline"),
        ("Can recommendation algorithms increase catalogue coverage?", "Experiment Lab · coverage & exposure Gini"),
        ("Can we reduce repetitive recommendations?", "Experiment Lab · repetition metrics"),
        ("Can lesser-known artists be recommended without much loss of relevance?", "Experiment Lab · novelty sweep"),
        ("How do strategies compare?", "Experiment Lab · strategy comparison"),
        ("Can users understand why a song was recommended?", "Why This Song? · Pilot Study"),
    ]
    st.markdown('<ul class="sb-reasons">' + "".join(f"<li><b>{q}</b><br><span style='font-size:13px'>→ {w}</span></li>" for q, w in rq)
                + "</ul>", unsafe_allow_html=True)

st.markdown("""
### Data
* **Spotify Tracks Dataset** (Kaggle, `maharshipandya/-spotify-tracks-dataset`): 114,000 rows, 114 genre labels, audio
  features retrieved from Spotify's Web API by the dataset author. Downloaded from public mirrors that serve a
  byte-identical file (SHA-256 checked). After cleaning: 79,868 unique songs.
* No Spotify scraping, no private listening data. Spotify features are optional: track links and the official embed
  player (only when switched on), plus album artwork if you provide your own API credentials.

### Ethics & limitations
* **Popularity is an exposure proxy**, never a quality score. "Lesser-known" always means in this dataset.
* **Audio features are estimates.** Valence and energy describe signal properties, not anyone's feelings.
* **This dataset is a sample.** It cannot measure the full Spotify catalogue or industry-wide exposure.
* **Relevance without users is a proxy.** Only the Feedback Dashboard and Pilot Study measure real relevance, and
  they report their actual (possibly zero) sample sizes.
* **No claims about artists** being independent, emerging or undiscovered — there is no metadata to support that.

SoundBridge AI is not affiliated with or endorsed by Spotify. It does not reproduce any proprietary recommendation
algorithm.
""")
ui.footer()
