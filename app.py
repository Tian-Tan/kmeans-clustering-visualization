from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from algorithm import KMeans  # Assuming the modified KMeans class is in algorithm.py

app = Flask(__name__)

# Global variables to store the current dataset, centroids, and KMeans object
current_data = None
current_centroids = None
kmeans = None

# Generate random dataset for KMeans
def generate_random_data():
    return np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(200)])

# Create Plotly plot based on the data and centroids
def create_plot(data, assignment=None, centers=None):
    trace_data = go.Scatter(
        x=[point[0] for point in data],
        y=[point[1] for point in data],
        mode='markers',
        marker=dict(
            color=assignment if assignment is not None else 'rgba(0, 123, 255, 0.6)',
            size=10
        ),
        name='Data Points'
    )

    traces = [trace_data]

    # Plot centroids if provided
    if centers is not None and len(centers) > 0:
        trace_centroids = go.Scatter(
            x=[center[0] for center in centers],
            y=[center[1] for center in centers],
            mode='markers',
            marker=dict(
                color='red',
                symbol='x',
                size=12,
                line=dict(width=2)
            ),
            name='Centroids'
        )
        traces.append(trace_centroids)

    layout = go.Layout(
        title='KMeans Clustering Data with Centroids',
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
        height=500,
        width=700
    )

    fig = go.Figure(data=traces, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/')
def index():
    global current_data
    if current_data is None:
        current_data = generate_random_data()
    plot = create_plot(current_data)
    return render_template('index.html', plot=plot)

@app.route('/generate-data', methods=['POST'])
def generate_data():
    global current_data, current_centroids, kmeans
    num_clusters = int(request.json.get('numClusters', 3))  # Default to 3 clusters if not provided

    # Generate new dataset
    current_data = generate_random_data()

    # Initialize KMeans object with new data and num_clusters
    kmeans = KMeans(current_data, num_clusters)

    # Create plot with the initial centroids
    plot = create_plot(current_data)
    return jsonify(plot=plot)

@app.route('/step-kmeans', methods=['POST'])
def step_kmeans():
    global kmeans, current_centroids
    if kmeans is None:
        return jsonify(error="KMeans not initialized"), 400

    # Perform one step of the KMeans algorithm
    step_result = kmeans.step()
    if step_result is None:
        return jsonify(error="KMeans algorithm has converged"), 200

    current_centroids, assignment = step_result
    
    # Create the plot with updated centroids and assignments
    plot = create_plot(kmeans.data, assignment=assignment, centers=current_centroids)
    return jsonify(plot=plot)

if __name__ == '__main__':
    app.run(debug=True)
