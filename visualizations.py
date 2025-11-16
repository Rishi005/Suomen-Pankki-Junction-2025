import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import pickle
import os
import pandas as pd
import networkx as nx

def create_interactive_network(graph, documents, overlaps, contradictions):
    """
    Create an interactive network visualization showing relationships
    """
    # Extract node positions using spring layout
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Create edges
    edge_traces = []
    
    # Overlap edges (green)
    overlap_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') == 'overlap']
    overlap_x = []
    overlap_y = []
    for edge in overlap_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        overlap_x.extend([x0, x1, None])
        overlap_y.extend([y0, y1, None])
    
    edge_traces.append(go.Scatter(
        x=overlap_x, y=overlap_y,
        line=dict(width=2, color='green'),
        hoverinfo='none',
        mode='lines',
        name='Overlaps'
    ))
    
    # Contradiction edges (red)
    contradiction_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('type') == 'contradiction']
    contradiction_x = []
    contradiction_y = []
    for edge in contradiction_edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        contradiction_x.extend([x0, x1, None])
        contradiction_y.extend([y0, y1, None])
    
    edge_traces.append(go.Scatter(
        x=contradiction_x, y=contradiction_y,
        line=dict(width=3, color='red'),
        hoverinfo='none',
        mode='lines',
        name='Contradictions'
    ))
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    # Color mapping for risk categories
    risk_colors = {
        'credit risk': '#FF6B6B',
        'market risk': '#4ECDC4', 
        'liquidity risk': '#45B7D1',
        'operational risk': '#96CEB4',
        'cluster_0': '#FECA57',
        'cluster_1': '#FF9FF3',
        'cluster_2': '#54A0FF'
    }
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Find document data
        doc_data = next((d for d in documents if d['id'] == node), {})
        risk = doc_data.get('risk_category', 'Unknown')
        requirements_count = len(doc_data.get('requirements', []))
        
        node_text.append(f"Document: {node}<br>Risk: {risk}<br>Requirements: {requirements_count}")
        node_color.append(risk_colors.get(risk, '#95AFC0'))
        node_size.append(10 + min(requirements_count * 2, 20))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='darkgray')
        ),
        name='Documents'
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title='Regulatory Document Network - Overlaps & Contradictions',
                    #    titlefont_size=16,
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Green lines: Overlapping requirements<br>Red lines: Contradictory requirements",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='black', size=10)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def create_risk_analysis_dashboard(documents, overlaps, contradictions):
    """
    Create a comprehensive dashboard showing risk analysis
    """
    # Prepare data
    risk_counts = {}
    cross_lingual_overlaps = [o for o in overlaps if o.get('cross_lingual', False)]
    
    for doc in documents:
        risk = doc.get('risk_category', 'Unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk Category Distribution', 
            'Overlaps vs Contradictions by Risk',
            'Cross-Lingual Overlaps',
            'Requirements per Document'
        ),
        # specs=[[{"type": "pie"}, {"type": "bar"}],
        #        [{"type": "bar"}, {"type": "histogram"}]]
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [None, None]]
    )
    
    # Pie chart - Risk distribution
    fig.add_trace(
        go.Pie(
            labels=list(risk_counts.keys()),
            values=list(risk_counts.values()),
            name="Risk Distribution"
        ),
        row=1, col=1
    )
    
    # Bar chart - Overlaps and contradictions by risk
    risk_overlaps = {}
    risk_contradictions = {}
    
    for overlap in overlaps:
        doc1_risk = next((d['risk_category'] for d in documents if d['id'] == overlap['doc1']), 'Unknown')
        risk_overlaps[doc1_risk] = risk_overlaps.get(doc1_risk, 0) + 1
    
    for contradiction in contradictions:
        doc1_risk = next((d['risk_category'] for d in documents if d['id'] == contradiction['doc1']), 'Unknown')
        risk_contradictions[doc1_risk] = risk_contradictions.get(doc1_risk, 0) + 1
    
    risks = list(set(list(risk_overlaps.keys()) + list(risk_contradictions.keys())))
    
    fig.add_trace(
        go.Bar(x=risks, y=[risk_overlaps.get(r, 0) for r in risks], name='Overlaps'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=risks, y=[risk_contradictions.get(r, 0) for r in risks], name='Contradictions'),
        row=1, col=2
    )
    
    # # Cross-lingual overlaps
    # language_pairs = {}
    # for overlap in cross_lingual_overlaps:
    #     pair = overlap.get('languages', 'Unknown')
    #     language_pairs[pair] = language_pairs.get(pair, 0) + 1
    
    # fig.add_trace(
    #     go.Bar(x=list(language_pairs.keys()), y=list(language_pairs.values()), 
    #            name='Cross-lingual Overlaps'),
    #     row=2, col=1
    # )
    
    # # Requirements distribution
    # req_counts = [len(doc.get('requirements', [])) for doc in documents]
    # fig.add_trace(
    #     go.Histogram(x=req_counts, nbinsx=20, name='Requirements per Document'),
    #     row=2, col=2
    # )
    
    # fig.update_layout(height=800, showlegend=True, title_text="Regulatory Risk Analysis Dashboard")
    return fig



####################################################################################################################################################
####################################################################################################################################################

def create_similarity_heatmap(overlaps, documents):
    """
    Create a heatmap showing similarity between documents based on overlaps
    """
    # Create similarity matrix
    doc_ids = list(set([d['id'] for d in documents]))
    similarity_matrix = np.zeros((len(doc_ids), len(doc_ids)))
    
    # Fill similarity matrix
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    for overlap in overlaps:
        i = doc_to_idx[overlap['doc1']]
        j = doc_to_idx[overlap['doc2']]
        similarity_matrix[i, j] = overlap['similarity']
        similarity_matrix[j, i] = overlap['similarity']
    
    # Add diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Get risk categories for labels
    risk_labels = []
    for doc_id in doc_ids:
        doc = next((d for d in documents if d['id'] == doc_id), {})
        risk_labels.append(doc.get('risk_category', 'Unknown'))
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=doc_ids,
        y=doc_ids,
        colorscale='Viridis',
        hoverongaps=False,
        text=[[f"Similarity: {sim:.3f}" for sim in row] for row in similarity_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Document Similarity Heatmap',
        xaxis_title='Documents',
        yaxis_title='Documents',
        height=600,
        width=800
    )
    
    return fig

####################################################################################################################################################
####################################################################################################################################################


def create_regulatory_timeline(documents, overlaps, contradictions):
    """
    Create a timeline showing when regulations were created/updated
    and their relationships over time
    """
    # Extract dates from filenames or metadata (you'll need to adapt this)
    timeline_data = []
    
    for doc in documents:
        # Try to extract date from filename or use modification time
        filename = doc['filename']
        # Simple heuristic - look for dates in filename
        import re
        date_match = re.search(r'(\d{4})[-_]?(\d{2})?[-_]?(\d{2})?', filename)
        
        if date_match:
            year = int(date_match.group(1))
            month = int(date_match.group(2)) if date_match.group(2) else 1
            day = int(date_match.group(3)) if date_match.group(3) else 1
        else:
            # Fallback: use file modification time
            file_path = f"path_to_files/{filename}"  # Update this path
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                year = datetime.fromtimestamp(mod_time).year
                month = datetime.fromtimestamp(mod_time).month
                day = datetime.fromtimestamp(mod_time).day
            else:
                year = 2020  # Default
                month = 1
                day = 1
        
        timeline_data.append({
            'Document': doc['id'],
            'Date': f"{year}-{month:02d}-{day:02d}",
            'Risk Category': doc.get('risk_category', 'Unknown'),
            'Requirements Count': len(doc.get('requirements', [])),
            'Overlaps': len([o for o in overlaps if o['doc1'] == doc['id'] or o['doc2'] == doc['id']]),
            'Contradictions': len([c for c in contradictions if c['doc1'] == doc['id'] or c['doc2'] == doc['id']])
        })
    
    df = pd.DataFrame(timeline_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    fig = px.scatter(df, x='Date', y='Risk Category', 
                     size='Requirements Count',
                     color='Risk Category',
                     hover_data=['Document', 'Overlaps', 'Contradictions'],
                     title='Regulatory Document Timeline')
    
    return fig

####################################################################################################################################################
####################################################################################################################################################


def create_all_visualizations(documents, overlaps, contradictions, graph):
    """
    Create all visualizations and save them as HTML files
    """
    print("🔄 Creating visualizations...")
    
    # 1. Interactive Network
    network_fig = create_interactive_network(graph, documents, overlaps, contradictions)
    network_fig.write_html("network_visualization.html")
    
    # 2. Risk Analysis Dashboard
    dashboard_fig = create_risk_analysis_dashboard(documents, overlaps, contradictions)
    dashboard_fig.write_html("risk_dashboard.html")
    
    # 3. Similarity Heatmap
    heatmap_fig = create_similarity_heatmap(overlaps, documents)
    heatmap_fig.write_html("similarity_heatmap.html")
    
    # # 4. Cross-Lingual Bridge
    # bridge_fig = create_cross_lingual_bridge(documents, overlaps)
    # bridge_fig.write_html("cross_lingual_bridge.html")
    
    # 5. Create a summary report
    create_summary_report(documents, overlaps, contradictions)
    
    print("✅ All visualizations created and saved as HTML files!")

def create_summary_report(documents, overlaps, contradictions):
    """
    Create a text summary report with key insights
    """
    total_docs = len(documents)
    total_overlaps = len(overlaps)
    total_contradictions = len(contradictions)
    
    cross_lingual_count = len([o for o in overlaps if o.get('cross_lingual', False)])
    
    risk_distribution = {}
    for doc in documents:
        risk = doc.get('risk_category', 'Unknown')
        risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
    
    # Create report
    report = f"""
    REGULATORY ANALYSIS REPORT
    =========================
    
    Documents Analyzed: {total_docs}
    Total Overlaps Found: {total_overlaps}
    Total Contradictions Found: {total_contradictions}
    Cross-Lingual Matches: {cross_lingual_count}
    
    Risk Distribution:
    {json.dumps(risk_distribution, indent=2)}
    
    KEY INSIGHTS:
    """
    
    # Add insights based on data
    if cross_lingual_count > 0:
        report += f"- Found {cross_lingual_count} cross-lingual regulatory matches between Finnish and English\n"
    
    if total_contradictions > 0:
        report += f"- ⚠️  Found {total_contradictions} potential regulatory contradictions that need review\n"
    
    # Find risk category with most overlaps
    if risk_distribution:
        max_risk = max(risk_distribution.items(), key=lambda x: x[1])
        report += f"- '{max_risk[0]}' is the most common risk category ({max_risk[1]} documents)\n"
    
    with open("analysis_report.txt", "w") as f:
        f.write(report)

####################################################################################################################################################
####################################################################################################################################################




import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

def create_3d_interactive_network(graph, documents, overlaps, contradictions):
    """
    Create an interactive 3D network visualization with enhanced risk analysis features
    """
    print("🔄 Creating 3D interactive network visualization...")
    
    # Create a 3D layout using force-directed algorithm or dimensionality reduction
    pos_3d = create_3d_layout(graph, documents)
    
    # Create edge traces for different relationship types
    edge_traces = create_3d_edge_traces(graph, pos_3d, overlaps, contradictions)
    
    # Create node trace with enhanced risk information
    node_trace = create_3d_node_trace(graph, documents, pos_3d)
    
    # Create cluster centroids for better visualization
    centroid_trace = create_cluster_centroids(documents, pos_3d)
    
    # Create the 3D figure
    fig = go.Figure(data=edge_traces + [node_trace] + [centroid_trace],
                   layout=create_3d_layout_config())
    
    # Add risk category legend
    fig = add_3d_legend(fig)
    
    print("✅ 3D network visualization created!")
    return fig

def create_3d_layout(graph, documents):
    """
    Create 3D positions using a combination of clustering and force-directed layout
    """
    # First get 2D positions using spring layout
    pos_2d = nx.spring_layout(graph, k=3, iterations=100, dim=2)
    
    # Convert to 3D by using risk categories and cluster information for Z-axis
    pos_3d = {}
    
    # Define risk hierarchy for Z-axis (you can customize this based on your risk priorities)
    risk_hierarchy = {
        'credit risk': 4,
        'market risk': 3, 
        'liquidity risk': 2,
        'operational risk': 1,
        'compliance risk': 0,
        'unknown': 0
    }
    
    for node in graph.nodes():
        x, y = pos_2d[node]
        
        # Find document data
        doc_data = next((d for d in documents if d['id'] == node), {})
        
        # Use risk category for Z-axis, cluster for variation
        risk = doc_data.get('risk_category', 'unknown').lower()
        cluster = doc_data.get('cluster', 0)
        
        # Base Z on risk hierarchy with some cluster-based variation
        z_base = risk_hierarchy.get(risk, 0)
        z_variation = (cluster % 3) * 0.3  # Small variation based on cluster
        z = z_base + z_variation
        
        pos_3d[node] = (x, y, z)
    
    return pos_3d

def create_3d_edge_traces(graph, pos_3d, overlaps, contradictions):
    """
    Create 3D edge traces for different relationship types
    """
    edge_traces = []
    
    # Overlap edges (green with varying opacity based on similarity)
    overlap_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('type') == 'overlap']
    
    for u, v, data in overlap_edges:
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        
        similarity = data.get('similarity', 0.5)
        opacity = max(0.3, similarity)  # Higher similarity = more opaque
        
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(
                color=f'rgba(0, 255, 0, {opacity})',
                width=2 + similarity * 4  # Thicker lines for higher similarity
            ),
            hoverinfo='text',
            text=f"Overlap: {similarity:.3f}",
            name='Overlaps',
            showlegend=False
        ))
    
    # Contradiction edges (red with pulsing effect)
    contradiction_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('type') == 'contradiction']
    
    for u, v, data in contradiction_edges:
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(
                color='rgba(255, 0, 0, 0.8)',
                width=4
            ),
            hoverinfo='text',
            text="CONTRADICTION",
            name='Contradictions',
            showlegend=False
        ))
    
    return edge_traces

def create_3d_node_trace(graph, documents, pos_3d):
    """
    Create enhanced 3D node trace with risk-based coloring and sizing
    """
    node_x, node_y, node_z = [], [], []
    node_text, node_color, node_size = [], [], []
    
    # Enhanced color mapping with gradients for each risk type
    risk_colors = {
        'credit risk': ['#FF6B6B', '#FF8E8E', '#FFB1B1'],  # Red gradient
        'market risk': ['#4ECDC4', '#6ED7CF', '#8EE1DA'],   # Teal gradient  
        'liquidity risk': ['#45B7D1', '#67C5D9', '#89D3E1'], # Blue gradient
        'operational risk': ['#96CEB4', '#ABD8C1', '#C0E2CE'], # Green gradient
        'compliance risk': ['#C44569', '#CC5F7E', '#D47993'], # Pink gradient
        'unknown': ['#95A5A6', '#ACB9B9', '#C3CDCD']         # Gray gradient
    }
    
    for node in graph.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        # Find document data
        doc_data = next((d for d in documents if d['id'] == node), {})
        risk = doc_data.get('risk_category', 'unknown').lower()
        requirements_count = len(doc_data.get('requirements', []))
        cluster = doc_data.get('cluster', 0)
        language = doc_data.get('language', 'unknown')
        
        # Create enhanced hover text
        hover_text = (
            f"<b>Document:</b> {node}<br>"
            f"<b>Risk:</b> {risk}<br>"
            f"<b>Requirements:</b> {requirements_count}<br>"
            f"<b>Cluster:</b> {cluster}<br>"
            f"<b>Language:</b> {language}<br>"
            f"<b>Position:</b> ({x:.2f}, {y:.2f}, {z:.2f})"
        )
        node_text.append(hover_text)
        
        # Color based on risk with gradient based on cluster
        base_colors = risk_colors.get(risk, risk_colors['unknown'])
        color_idx = cluster % len(base_colors)
        node_color.append(base_colors[color_idx])
        
        # Size based on number of requirements with minimum size
        node_size.append(max(8, 5 + requirements_count * 1.5))
    
    # Create the node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            opacity=0.9,
            line=dict(width=2, color='darkgray')
        ),
        text=node_text,
        hoverinfo='text',
        name='Regulatory Documents',
        customdata=[node for node in graph.nodes()]  # Store node IDs for interaction
    )
    
    return node_trace

def create_cluster_centroids(documents, pos_3d):
    """
    Create transparent centroid markers for each cluster
    """
    clusters = set(doc.get('cluster', 0) for doc in documents)
    
    centroid_x, centroid_y, centroid_z = [], [], []
    centroid_text = []
    
    for cluster in clusters:
        cluster_docs = [doc for doc in documents if doc.get('cluster') == cluster]
        if not cluster_docs:
            continue
            
        # Calculate centroid position
        cluster_positions = [pos_3d[doc['id']] for doc in cluster_docs if doc['id'] in pos_3d]
        if not cluster_positions:
            continue
            
        centroid = np.mean(cluster_positions, axis=0)
        centroid_x.append(centroid[0])
        centroid_y.append(centroid[1])
        centroid_z.append(centroid[2])
        
        # Get cluster risk profile
        risks = [doc.get('risk_category', 'unknown') for doc in cluster_docs]
        most_common_risk = max(set(risks), key=risks.count) if risks else 'unknown'
        
        centroid_text.append(f"Cluster {cluster}<br>Main Risk: {most_common_risk}<br>Documents: {len(cluster_docs)}")
    
    centroid_trace = go.Scatter3d(
        x=centroid_x, y=centroid_y, z=centroid_z,
        mode='markers',
        marker=dict(
            size=15,
            color='rgba(255, 255, 255, 0.8)',
            symbol='diamond',
            line=dict(width=3, color='black')
        ),
        text=centroid_text,
        hoverinfo='text',
        name='Cluster Centroids',
        opacity=0.7
    )
    
    return centroid_trace

def create_3d_layout_config():
    """
    Create configuration for the 3D layout
    """
    layout = go.Layout(
        title=dict(
            text='<b>3D Regulatory Risk Network Analysis</b><br><sub>Node Color = Risk Type | Node Size = Requirements Count | Z-axis = Risk Hierarchy</sub>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                title='X - Semantic Similarity',
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                backgroundcolor='rgba(240, 240, 240, 0.1)'
            ),
            yaxis=dict(
                title='Y - Document Relationships', 
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                backgroundcolor='rgba(240, 240, 240, 0.1)'
            ),
            zaxis=dict(
                title='Z - Risk Hierarchy<br>(Credit > Market > Liquidity > Operational)',
                showgrid=True,
                zeroline=True,
                showticklabels=True,
                backgroundcolor='rgba(240, 240, 240, 0.1)'
            ),
            bgcolor='rgba(250, 250, 250, 1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, b=0, t=100),
        annotations=[
            dict(
                text="💡 <b>Interact:</b> Drag to rotate | Scroll to zoom | Hover for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color='gray')
            )
        ]
    )
    return layout

def add_3d_legend(fig):
    """
    Add a comprehensive legend for the 3D visualization
    """
    # Risk type legend markers
    risk_types = ['credit risk', 'market risk', 'liquidity risk', 'operational risk', 'compliance risk']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#C44569']
    
    for risk, color in zip(risk_types, colors):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=risk,
            showlegend=True
        ))
    
    # Relationship type legend
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines',
        line=dict(color='green', width=3),
        name='Overlaps (thicker = more similar)',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='lines', 
        line=dict(color='red', width=4),
        name='Contradictions',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=15, color='white', symbol='diamond', line=dict(width=2, color='black')),
        name='Cluster Centroids',
        showlegend=True
    ))
    
    return fig

def create_animated_3d_network(graph, documents, overlaps, contradictions):
    """
    Create an animated 3D network that shows risk evolution or relationships over time
    """
    print("🎬 Creating animated 3D network...")
    
    frames = []
    
    # Create multiple perspectives for animation
    camera_positions = [
        dict(eye=dict(x=1.5, y=1.5, z=1.5)),  # Default
        dict(eye=dict(x=0.0, y=2.0, z=0.0)),  # Top view
        dict(eye=dict(x=2.0, y=0.0, z=0.0)),  # Side view
        dict(eye=dict(x=0.0, y=0.0, z=2.5)),  # Front view
    ]
    
    for i, camera in enumerate(camera_positions):
        fig_frame = create_3d_interactive_network(graph, documents, overlaps, contradictions)
        fig_frame.layout.scene.camera.eye = camera['eye']
        frames.append(go.Frame(data=fig_frame.data, layout=fig_frame.layout, name=f"frame_{i}"))
    
    # Create base figure
    base_fig = create_3d_interactive_network(graph, documents, overlaps, contradictions)
    
    # Add animation controls
    base_fig.frames = frames
    base_fig.layout.updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(label="Play",
                 method="animate",
                 args=[None, {"frame": {"duration": 1000, "redraw": True}, 
                             "fromcurrent": True}]),
            dict(label="Pause",
                 method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, 
                              "mode": "immediate",
                              "transition": {"duration": 0}}])
        ]
    )]
    
    return base_fig

# Enhanced usage in your pipeline:
def create_advanced_risk_visualizations(documents, overlaps, contradictions, graph):
    """
    Create comprehensive 3D visualizations for risk analysis
    """
    print("🚀 Creating advanced 3D risk visualizations...")
    
    # 1. Main 3D interactive network
    fig_3d = create_3d_interactive_network(graph, documents, overlaps, contradictions)
    fig_3d.write_html("3d_risk_network.html")
    print("✅ 3D network saved as '3d_risk_network.html'")
    
    # 2. Animated version (optional)
    try:
        fig_animated = create_animated_3d_network(graph, documents, overlaps, contradictions)
        fig_animated.write_html("3d_animated_network.html")
        print("✅ Animated 3D network saved as '3d_animated_network.html'")
    except Exception as e:
        print(f"⚠️  Animated network failed: {e}")
    
    # 3. Risk-focused subgraphs
    create_risk_specific_networks(documents, overlaps, contradictions, graph)
    
    return fig_3d

def create_risk_specific_networks(documents, overlaps, contradictions, graph):
    """
    Create separate 3D networks for each major risk category
    """
    risk_categories = set(doc.get('risk_category', 'unknown') for doc in documents)
    
    for risk in risk_categories:
        if risk == 'unknown':
            continue
            
        # Filter documents for this risk
        risk_docs = [doc for doc in documents if doc.get('risk_category') == risk]
        risk_doc_ids = set(doc['id'] for doc in risk_docs)
        
        # Create subgraph for this risk
        subgraph = graph.subgraph(risk_doc_ids)
        
        # Filter overlaps and contradictions for this risk
        risk_overlaps = [o for o in overlaps if o['doc1'] in risk_doc_ids and o['doc2'] in risk_doc_ids]
        risk_contradictions = [c for c in contradictions if c['doc1'] in risk_doc_ids and c['doc2'] in risk_doc_ids]
        
        if len(risk_docs) > 1:  # Only create if we have multiple documents
            try:
                risk_fig = create_3d_interactive_network(subgraph, risk_docs, risk_overlaps, risk_contradictions)
                risk_fig.update_layout(title=f"3D Network: {risk.upper()} Risk")
                risk_fig.write_html(f"3d_network_{risk.lower().replace(' ', '_')}.html")
                print(f"✅ {risk} risk network saved")
            except Exception as e:
                print(f"⚠️  Failed to create {risk} network: {e}")



####################################################################################################################################################
####################################################################################################################################################



documents_path = "/home/rishika/Suomen-Pankki-Junction-2025/clean1_processed_documents.json"
relationships_path = "/home/rishika/Suomen-Pankki-Junction-2025/clean1_relationships.json"


with open(documents_path, "r", encoding="utf-8") as d:
    documents = json.load(d)
with open(relationships_path, "r", encoding="utf-8") as d:
    relationships = json.load(d)

overlaps = relationships['overlaps']
contradictions = relationships['contradictions']

with open("/home/rishika/Suomen-Pankki-Junction-2025/clean1_regulation_graph.pickle", "rb") as f:
    G = pickle.load(f)

# Create visualizations
create_all_visualizations(documents, overlaps, contradictions, G)
create_advanced_risk_visualizations(documents, overlaps, contradictions, G)
