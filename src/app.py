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
                {'label': 'Left Elbow Angle', 'value': 'Left Elbow Angle'},
                {'label': 'Right Elbow Angle', 'value': 'Right Elbow Angle'},
                {'label': 'Left Knee Angle', 'value': 'Left Knee Angle'},
                {'label': 'Right Knee Angle', 'value': 'Right Knee Angle'}
            ],
            value=['Wrist', 'Left Elbow Angle'],  # default plots shown on screen
            multi=True,
            className='dropdown-bar'
        )),
    ],),
    dbc.Row(id='graph-row'),
    dbc.Row([
        dbc.Col(html.Div(html.Img(id='live-video', style={'height': VIDEO_HEIGHT, 'width': VIDEO_WIDTH, 'display': 'block', 'margin': 'auto'}))),
    ]),
    # Center the switch in the middle of the screen
    dbc.Row([
        dbc.Col(
            dbc.Switch(
                id='keypoint-switch',
                label="Keypoints",
                value=False,  # Default is off
                className="mb-3"
            ),
            width="auto",  # Automatically adjust column width
            style={'textAlign': 'center'}  # Center within the column
        ),
    ], justify="center", className="mb-4"),  # Center the row and add margin below
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
    window_size = 5  # Adjust this size to control the smoothing effect
    data['Left Wrist'] = smooth_data(data['Left Wrist'], window_size)
    data['Right Wrist'] = smooth_data(data['Right Wrist'], window_size)
    data['Left Elbow'] = smooth_data(data['Left Elbow'], window_size)
    data['Right Elbow'] = smooth_data(data['Right Elbow'], window_size)
    data['Left Hip'] = smooth_data(data['Left Hip'], window_size)
    data['Right Hip'] = smooth_data(data['Right Hip'], window_size)
    data['Left Knee'] = smooth_data(data['Left Knee'], window_size)
    data['Right Knee'] = smooth_data(data['Right Knee'], window_size)

    x = np.arange(len(data))

    figures = {
        'Wrist': {
            'left': go.Scatter(x=x, y=data['Left Wrist'], mode='lines', name='Left Wrist', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['Right Wrist'], mode='lines', name='Right Wrist', line=dict(width=1)),
            'title': 'Wrist Movements'
        },
        'Elbow': {
            'left': go.Scatter(x=x, y=data['Left Elbow'], mode='lines', name='Left Elbow', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['Right Elbow'], mode='lines', name='Right Elbow', line=dict(width=1)),
            'title': 'Elbow Movements'
        },
        'Hip': {
            'left': go.Scatter(x=x, y=data['Left Hip'], mode='lines', name='Left Hip', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['Right Hip'], mode='lines', name='Right Hip', line=dict(width=1)),
            'title': 'Hip Movements'
        },
        'Knee': {
            'left': go.Scatter(x=x, y=data['Left Knee'], mode='lines', name='Left Knee', line=dict(width=1)),
            'right': go.Scatter(x=x, y=data['Right Knee'], mode='lines', name='Right Knee', line=dict(width=1)),
            'title': 'Knee Movements'
        },
        'Left Elbow Angle': {
            'angle': go.Scatter(x=x, y=data['Left Elbow Angle'], mode='lines', name='Left Elbow Angle', line=dict(width=1)),
            'title': 'Left Elbow Angle'
        },
        'Right Elbow Angle': {
            'angle': go.Scatter(x=x, y=data['Right Elbow Angle'], mode='lines', name='Right Elbow Angle', line=dict(width=1)),
            'title': 'Right Elbow Angle'
        },
        'Left Knee Angle': {
            'angle': go.Scatter(x=x, y=data['Left Knee Angle'], mode='lines', name='Left Knee Angle', line=dict(width=1)),
            'title': 'Left Knee Angle'
        },
        'Right Knee Angle': {
            'angle': go.Scatter(x=x, y=data['Right Knee Angle'], mode='lines', name='Right Knee Angle', line=dict(width=1)),
            'title': 'Right Knee Angle'
        }
    }

    graph_components = []
    for graph_type in selected_graphs:
        if graph_type in figures:
            fig = go.Figure()
            if 'left' in figures[graph_type]:
                fig.add_trace(figures[graph_type]['left'])
            if 'right' in figures[graph_type]:
                fig.add_trace(figures[graph_type]['right'])
            if 'angle' in figures[graph_type]:
                fig.add_trace(figures[graph_type]['angle'])
            fig.update_layout(
                title=figures[graph_type]['title'], 
                xaxis_title='Frame Index', 
                yaxis_title='Pixel Position',
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
