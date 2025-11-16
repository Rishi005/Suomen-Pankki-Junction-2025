# Running the visualisations
In order to see the visualisations and interact with the different nodes, run visualisations.py and then view_visualisations.py


# General info about the project

The program identifies and categorises the key risks (e.g. operational risk, credit risk, liquidity risk) addressed by legislation involving financial corporations granting credit. It compares input legislation strings to identify conflicting specifications in legislation addressing these risks and returns overlapping or contradicting legislation.
The implementation allows comparison of legislation in different languages by transforming all of the requirements from the cleaned JSONs into embeddings using Gemini's text-embedding-004, which can capture semantic meaning in multiple languages. This was needed as input legislation compared Finnish and English legislative documents. Agglomerative Clustering was then used to cluster these embeddings with cosine similarity to capture the natural groupings of both Finnish and English requirements together​.

Categories of risks are generated to best fit these clusters and are assigned to each requirement. Similar requirements (when looking at similarity in meaning) should fall under the same category of risk​. FAISS was used to find overlapping samples for efficient similarity search, by extracting the most similar pairs of requirements by thresholding the similarity distance at 0.9: ​as overlapping samples should have close to the same meaning, with our embeddings it should return a close to 1 cos similarity.

Contradicting samples were identified by subsampling less similar overlapping samples and utilizing Gemini's reasoning skills to determine if two policies are defined by different specifications​. Contradicting samples will be similar in meaning but differ in a small aspect of their specification such as different procedural rules​.

A NetworkX graph is then built where the nodes and metadata (e.g. risk category) are the requirements and the edges are if contradicting or overlapping​.

