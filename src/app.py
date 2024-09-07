import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import threading
import cv2
import base64
import numpy as np
import dash_bootstrap_components as dbc
from yolov8_utils import write_pose_video

CSV_FILE = 'keypoints.csv'
VIDEO_FILE = 'DJI_0886.MP4'

external_stylesheets = [dbc.themes.BOOTSTRAP, '/assets/styles.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

VIDEO_HEIGHT = 360  
VIDEO_WIDTH = 640

app.layout = html.Div([
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='graph-dropdown',
            options=[
                {'label': 'Wrist', 'value': 'Wrist'},
                {'label': 'Elbow Angles', 'value': 'Elbow Angles'},  # Combined Elbow Angles
                {'label': 'Knee Angles', 'value': 'Knee Angles'}, 
                {'label': 'Nose', 'value': 'Nose'}
            ],
            value=['Nose', 'Elbow Angles', 'Knee Angles'],  # Default options selected
            multi=True,
            className='dropdown-bar'
        )),
    ],),
    dbc.Row(id='graph-row'),
    dbc.Row([
        dbc.Col(html.Div(html.Img(id='live-video', style={'height': VIDEO_HEIGHT, 'width': VIDEO_WIDTH, 'display': 'block', 'margin': 'auto'}))),
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Switch(
                id='keypoint-switch',
                label="Keypoints",
                value=False,
                className="mb-3"
            ),
            width="auto",
            style={'textAlign': 'center'}
        ),
    ], justify="center", className="mb-4"),
    dcc.Interval(
        id='interval-component',
        interval=100,  
        n_intervals=0
    )
])

current_frame = [None, None]  # Initialize to hold both frames (with and without keypoints)
frame_lock = threading.Lock()

# Start the thread to write the pose video
t1 = threading.Thread(target=write_pose_video, args=(VIDEO_FILE, CSV_FILE, frame_lock, current_frame))
t1.start()

def smooth_data(data, window_size=5):
    """Apply a moving average filter to smooth the data."""
    return data.rolling(window=window_size, min_periods=1).mean()

@app.callback(
    Output('graph-row', 'children'),
    [Input('graph-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)

def update_graph(selected_graphs, n_intervals):
    try:
        data = pd.read_csv(CSV_FILE)
    except Exception as e:
        return []

    # Apply smoothing
    window_size = 5
    data['Left Wrist'] = smooth_data(data['Left Wrist'], window_size)
    data['Right Wrist'] = smooth_data(data['Right Wrist'], window_size)
    data['Left Elbow'] = smooth_data(data['Left Elbow'], window_size)
    data['Right Elbow'] = smooth_data(data['Right Elbow'], window_size)
    data['Left Knee'] = smooth_data(data['Left Knee'], window_size)
    data['Right Knee'] = smooth_data(data['Right Knee'], window_size)
    data['Left Elbow Angle'] = smooth_data(data['Left Elbow Angle'], window_size)
    data['Right Elbow Angle'] = smooth_data(data['Right Elbow Angle'], window_size)
    data['Left Knee Angle'] = smooth_data(data['Left Knee Angle'], window_size)
    data['Right Knee Angle'] = smooth_data(data['Right Knee Angle'], window_size)
    data['Nose'] = smooth_data(data['Nose'], window_size)

    x = np.arange(len(data))

    # Combine left and right angles within the figures dictionary
    figures = {
        'Wrist': {
            'data': [
                go.Scatter(x=x, y=data['Left Wrist'], mode='lines', name='Left Wrist', line=dict(width=1)),
                go.Scatter(x=x, y=data['Right Wrist'], mode='lines', name='Right Wrist', line=dict(width=1))
            ],
            'title': 'Wrist Movements',
            'yaxis_title': 'Pixel Position'
        },
        'Elbow Angles': {
            'data': [
                go.Scatter(x=x, y=data['Left Elbow Angle'], mode='lines', name='Left Elbow Angle', line=dict(width=1)),
                go.Scatter(x=x, y=data['Right Elbow Angle'], mode='lines', name='Right Elbow Angle', line=dict(width=1))
            ],
            'title': 'Elbow Angles',
            'yaxis_title': 'Angle (in degrees)'  # Use degree label
        },
        'Knee Angles': {
            'data': [
                go.Scatter(x=x, y=data['Left Knee Angle'], mode='lines', name='Left Knee Angle', line=dict(width=1)),
                go.Scatter(x=x, y=data['Right Knee Angle'], mode='lines', name='Right Knee Angle', line=dict(width=1))
            ],
            'title': 'Knee Angles',
            'yaxis_title': 'Angle (in degrees)'  # Use degree label
        },
        'Nose': {
            'data': [
                go.Scatter(x=x, y=data['Nose'], mode='lines', name='Nose', line=dict(width=1))
            ],
            'title': 'Velocity',
            'yaxis_title': 'Speed (m/s)'
        }
    }

    # Generate graphs for the selected options
    graph_components = []
    for graph_type in selected_graphs:
        if graph_type in figures:
            fig = go.Figure(data=figures[graph_type]['data'])
            fig.update_layout(
                title=figures[graph_type]['title'],
                xaxis_title='Time',
                yaxis_title=figures[graph_type]['yaxis_title'],
                margin=dict(t=50, b=30, l=30, r=30),
                height=350
            )
            graph_components.append(dbc.Col(dcc.Graph(figure=fig), width=4))

    return graph_components


@app.callback(
    Output('live-video', 'src'),
    [Input('interval-component', 'n_intervals'),
    Input('keypoint-switch', 'value')]  # Switch Input
)
def update_video(n_intervals, show_keypoints):
    with frame_lock:
        if current_frame[0] is not None and current_frame[1] is not None:
            # Show the appropriate frame based on the switch state
            if show_keypoints:
                frame = current_frame[0]  # Frame with keypoints
            else:
                frame = current_frame[1]  # Frame without keypoints
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_src = 'data:image/jpg;base64,' + base64.b64encode(buffer).decode('utf-8')
        else:
            frame_src = ''
    return frame_src

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
