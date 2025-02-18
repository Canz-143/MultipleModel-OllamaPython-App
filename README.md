# Local LLM Interface

This project is a PyQt6-based application that allows users to interact with local language models (LLMs) and visualize CSV data using Plotly. The application provides a user-friendly interface for uploading CSV files, asking questions to the LLM, and creating various types of plots.

## Features

- Upload and preview CSV files
- Ask questions to the LLM with optional CSV context
- Create and display different types of plots (Bar Chart, Scatter Plot, Line Plot, Box Plot)
- User-friendly interface with tabs for chat and visualizations

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Canz-143/MultipleModel-OllamaPython-App.git
    cd Local-LLM-Interface
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```sh
    python app.py
    ```

2. Use the interface to:
    - Upload a CSV file
    - Ask questions to the LLM
    - Create and view plots

## Requirements

- Python 3.9+
- See `requirements.txt` for a list of required packages