# DeepVision - AI Video Threat Detector

DeepVision is an intelligent video surveillance solution powered by Google Gemini. It analyzes live video or recorded footage in real-time to detect potential security threats such as weapons and physical altercations, providing immediate visual and audio alerts via Google Cloud Text-to-Speech.

## Features

*   **Real-time Threat Detection:** Utilizes Google's Gemini 1.5 Flash model to analyze video frames for suspicious activities.
*   **Supports Multiple Sources:** Can process pre-recorded video files or capture live feed from a webcam.
*   **Audible Alerts:** Uses Google Cloud Text-to-Speech (gTTS) and `playsound` to notify operators of detected threats.
*   **Interactive Display:** Shows the video feed, current detection status (NORMAL/ABNORMAL), a description of the scene, and prominent visual cues for active alerts.
*   **Alert Acknowledgment:** Operators can acknowledge alerts (by pressing 'c'), stopping repeating audio notifications.
*   **Configurable:** Allows setting the frame processing interval, API key, and alert cooldown periods.
*   **Environment-Friendly:** Leverages `.env` files for secure API key management.

## Prerequisites

*   Python 3.8 or higher
*   A Google AI (Gemini) API Key
*   (For audio alerts) A working sound output. `playsound` might require additional system libraries depending on your OS:
    *   **Linux:** May require `mpg123` or `gstreamer` (`sudo apt-get install mpg123` or appropriate gstreamer packages).
    *   **macOS/Windows:** Often works out-of-the-box.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/Makobi5/DeepVision.git) # 
    ```

2.  **Create and activate a virtual environment (recommended):**
    The script is often run within a virtual environment named `myenv`.

    *   **For Windows:**
        ```bash
        python -m venv myenv
        myenv\Scripts\activate
        ```
    *   **For macOS/Linux:**
        ```bash
        python3 -m venv myenv
        source myenv/bin/activate
        ```

3.  **Install dependencies:**
    A `requirements.txt` file should be included in the repository.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not present, you can create it after manually installing the necessary packages (see below) in your virtual environment:
    ```bash
    pip freeze > requirements.txt
    ```
    Key dependencies include: `google-generativeai`, `gTTS`, `playsound`, `opencv-python`, `python-dotenv`, `Pillow`.

4.  **Set up your API Key:**
    Create a file named `.env` in the root project directory (alongside `detect.py`). Add your Google AI API Key to this file:
    ```env
    GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
    ```
    Replace `YOUR_ACTUAL_GEMINI_API_KEY` with your valid key.

## Usage

Once the setup is complete, you can run the `detect.py` script from your activated virtual environment (`myenv`).

**1. Analyzing a video file:**

Provide the path to the video file as a command-line argument.
```bash
python detect.py path/to/your/video.avi or --webcam to pick footage from webcam
