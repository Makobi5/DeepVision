#!/usr/bin/env python3
import argparse
import sys
import os
import time
from pathlib import Path
import cv2
import base64
import logging
import threading
import queue
import concurrent.futures
from collections import deque
from typing import Optional, Deque, Dict, Any
from dotenv import load_dotenv
import textwrap
import uuid
from gtts import gTTS
import platform

# Suppress specific warnings...
try:
    from google.protobuf.symbol_database import Default
    # logging.getLogger('google.protobuf.symbol_database').setLevel(logging.ERROR)
except ImportError:
    pass

# --- Import Google Cloud TTS ---
try:
    from google.cloud import texttospeech
except ImportError:
    print("Error: 'google-cloud-texttospeech' library not found.")
    texttospeech = None

try:
    from playsound import playsound
except ImportError:
    print("Error: 'playsound' library not found.")
    playsound = None

# --- Import Gemini and other libs ---
try:
    import google.generativeai as genai
    from PIL import Image
    import io
except ImportError:
    print("Error: Required libraries not found.")
    sys.exit(1)


# --- Configuration ---
CLASSIFICATION_HISTORY_LENGTH = 5
ABNORMAL_THRESHOLD_COUNT = 2
MAX_API_RETRIES = 3
API_RETRY_BASE_DELAY = 1.0
API_TIMEOUT = 25
MAX_WORKERS = 4
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
ALERT_COOLDOWN = 10
DEFAULT_ALERT_MESSAGE = "Attention: Abnormal event detected!"

# --- Constants for Display ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_STATUS = 0.8
FONT_SCALE_INFO = 0.6
FONT_THICKNESS_STATUS = 2
FONT_THICKNESS_INFO = 1
COLOR_NORMAL = (0, 255, 0)
COLOR_ABNORMAL = (0, 0, 255)
COLOR_INFO = (255, 255, 255)
BORDER_THICKNESS = 10
DESC_WRAP_WIDTH = 70


class GeminiVideoAnalyzer:
    def __init__(self, api_key: str, frame_interval: int = 30, alert_cooldown: int = ALERT_COOLDOWN):
        self.api_key = api_key
        self.frame_interval = max(1, frame_interval)
        self.gemini_model = None

        # Add these alert variables
        self.active_alert = False
        self.alert_acknowledge_needed = False
        self.alert_message = ""
        self.alert_repeat_interval = 5
        self.last_repeating_alert_time = 0

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.results_queue = queue.Queue()
        self.current_status_is_abnormal = False
        self.last_description = "Initializing..."
        self.previous_status_is_abnormal = False

        self.classification_history: Deque[bool] = deque(maxlen=CLASSIFICATION_HISTORY_LENGTH)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='GeminiWorker')
        self.futures: Deque[concurrent.futures.Future] = deque()

        # Initialize stop_event FIRST
        self.stop_event = threading.Event()

        # --- TTS Alert System ---
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0
        self.alerts_enabled = True
        self.temp_dir = Path('temp_audio_alerts')
        if not self.temp_dir.exists():
            self.temp_dir.mkdir()
        
        self.tts_queue = queue.Queue()
        
        # Start TTS thread
        self.tts_thread = threading.Thread(target=self._process_tts_queue, daemon=True, name="TTSProcessor")
        self.tts_thread.start()
        
        self.logger.info("Alert system initialized with sound notifications")

        # --- Configure Gemini API ---
        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            self.logger.info(f"Gemini API configured successfully using model: {GEMINI_MODEL_NAME}")
        except Exception as e:
            self.logger.error(f"Failed to configure Google AI API: {e}", exc_info=True)
            raise ValueError(f"Gemini API configuration failed: {e}") from e

    def _process_tts_queue(self):
        """Thread that processes TTS messages from the queue and handles repeating alerts."""
        while not self.stop_event.is_set():
            try:
                # Process regular TTS messages from queue
                if not self.tts_queue.empty():
                    message = self.tts_queue.get_nowait()
                    try:
                        filename = self.temp_dir / f"alert_{uuid.uuid4().hex}.mp3"
                        is_critical = message.startswith("ATTENTION! Potential crime scene")
                        
                        if is_critical:
                            tts = gTTS(text=message, lang='en', slow=False)
                        else:
                            tts = gTTS(text=message, lang='en')
                        
                        tts.save(str(filename))
                        if playsound:
                            playsound(str(filename))
                        
                        try:
                            filename.unlink()
                        except:
                            pass
                    except Exception as e:
                        self.logger.error(f"TTS Error: {e}")
                
                # Handle repeating alerts
                current_time = time.time()
                if self.alert_acknowledge_needed and self.alerts_enabled:
                    if (current_time - self.last_repeating_alert_time) > self.alert_repeat_interval:
                        self.last_repeating_alert_time = current_time
                        
                        repeat_message = f"ATTENTION! {self.alert_message} Please acknowledge this alert by pressing the 'c' key."
                        
                        try:
                            filename = self.temp_dir / f"repeat_alert_{uuid.uuid4().hex}.mp3"
                            tts = gTTS(text=repeat_message, lang='en', slow=False)
                            tts.save(str(filename))
                            if playsound:
                                playsound(str(filename))
                            try:
                                filename.unlink()
                            except:
                                pass
                        except Exception as e:
                            self.logger.error(f"Repeating TTS Error: {e}")
                            
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in TTS thread: {e}")
                
            # Brief sleep to prevent CPU overuse
            time.sleep(0.05)

    def trigger_alert(self, camera_name=None, immediate_description=None):
        """Trigger an alert and set it as active until acknowledged"""
        current_time = time.time()
        
        if ((current_time - self.last_alert_time) > self.alert_cooldown and 
                self.alerts_enabled and not self.alert_acknowledge_needed):
            
            self.last_alert_time = current_time
            self.last_repeating_alert_time = current_time
            
            description = immediate_description if immediate_description else self.last_description
            if not description:
                description = "Unknown event"
            
            if "." in description:
                brief_description = description.split('.')[0].strip()
            else:
                brief_description = description.strip()
                
            if len(brief_description) > 100:
                brief_description = brief_description[:97] + "..."
                
            alert_message = f"{brief_description}"
            if camera_name:
                alert_message = f"{brief_description} on {camera_name}"
            
            self.logger.info(f"ALERT: {alert_message}")
            self.alert_message = alert_message
            self.alert_acknowledge_needed = True
            
            initial_message = f"ATTENTION! Potential crime scene that needs attention is detected. {alert_message}. Please acknowledge this alert by pressing the 'c' key."
            self.tts_queue.put(initial_message)
            
            return True
        return False

    def acknowledge_alert(self):
        """Acknowledge and stop the current repeating alert"""
        if self.alert_acknowledge_needed:
            self.alert_acknowledge_needed = False
            self.logger.info(f"Alert acknowledged by operator: '{self.alert_message}'")
            self.tts_queue.put("Alert acknowledged. Thank you for your attention.")
            return True
        return False    

    def toggle_alerts(self):
        """Toggle alerts on/off"""
        self.alerts_enabled = not self.alerts_enabled
        status = "enabled" if self.alerts_enabled else "disabled"
        self.logger.info(f"Audio alerts {status}")
        return self.alerts_enabled

    def stop_tts(self):
        """Clean up TTS resources"""
        self.logger.info("Stopping TTS system...")
        self.alert_acknowledge_needed = False
        
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
        
        try:
            for file in self.temp_dir.glob("alert_*.mp3"):
                try:
                    file.unlink()
                except:
                    pass
            for file in self.temp_dir.glob("repeat_alert_*.mp3"):
                try:
                    file.unlink()
                except:
                    pass
            self.logger.info("Cleaned up temporary audio files")
        except Exception as e:
            self.logger.error(f"Error cleaning temp audio files: {e}")  

    def _cleanup_futures(self):
        """Clean up completed futures"""
        completed_futures = 0
        indices_to_remove = [i for i, f in enumerate(self.futures) if f.done()]
        for i in sorted(indices_to_remove, reverse=True):
            try:
                result = self.futures[i].result()
            except Exception as e:
                self.logger.error(f"Error in background Gemini task: {e}", exc_info=False)
            finally:
                del self.futures[i]
                completed_futures += 1
        
        if completed_futures > 0:
            self.logger.debug(f"Cleaned up {completed_futures} completed futures")

    def _parse_gemini_response(self, raw_text: str) -> (bool, str):
        """Parse Gemini response to extract classification and description"""
        lines = raw_text.strip().split('\n')
        is_abnormal = False
        description = "Parsing error or no description."

        if not lines:
            self.logger.warning("Received empty response from Gemini.")
            return False, "Empty API response."

        first_line_upper = lines[0].strip().upper()
        if first_line_upper == "ABNORMAL":
            is_abnormal = True
            if len(lines) > 1:
                description = "\n".join(lines[1:]).strip()
            else:
                description = "Abnormal event detected (no specific description)."
        elif first_line_upper == "NORMAL":
            is_abnormal = False
            if len(lines) > 1:
                description = "\n".join(lines[1:]).strip()
            else:
                description = "Normal scene (no specific description)."
        else:
            self.logger.warning(f"Gemini response did not start with NORMAL/ABNORMAL: '{lines[0]}'")
            description = raw_text.strip()
            if any(keyword in description.lower() for keyword in ["weapon", "fight", "altercation", "gun", "knife", "assault", "struggle", "attack", "running", "chased", "fallen", "agitated"]):
                is_abnormal = True
            else:
                is_abnormal = False

        if not description:
            description = "Abnormal event detected." if is_abnormal else "Normal scene."
        return is_abnormal, description

# In class GeminiVideoAnalyzer:

# In class GeminiVideoAnalyzer:

    def _make_api_request(self, frame_bytes: bytes, timestamp: float) -> Dict[str, Any]:
        """Make API request to Gemini with proper error handling"""
        if not self.gemini_model:
            self.logger.error("Gemini model not initialized.")
            return {"is_abnormal": False, "description": "Error: Model not initialized", "timestamp": timestamp}
        
        retry_count = 0
        
        # --- NEW "CAUSE-FIRST" HIERARCHICAL PROMPT ---
        prompt = """You are a hyper-vigilant AI security analyst. Your single most important job is to identify the **human actor** causing a threat and describe their action.

                You **MUST** follow this internal thought process step-by-step before giving an answer:
                **Step 1. Scan for Weapons:** Is there a person holding a gun, knife, or other weapon?
                    - If YES: Stop. Your entire analysis is now about this armed person. Formulate the description based ONLY on this.
                    - If NO: Proceed to Step 2.

                **Step 2. Scan for Physical Aggression:** Is there an active fight, a person assaulting another, or a clear aggressor?
                    - If YES: Stop. Your analysis is about the aggressors. Formulate the description based ONLY on this.
                    - If NO: Proceed to Step 3.

                **Step 3. Analyze the Scene for Critical Events:** If there are no visible weapons or fights, what is the most likely cause of any distress? (e.g., panicked running, people on the ground). Describe this secondary event.

                **Your Response Format is ABSOLUTELY STRICT:**

                **Line 1:** A single word: "ABNORMAL" or "NORMAL".
                - Classify as ABNORMAL if the result of your thought process (Step 1, 2, or 3) indicates a threat.

                **Line 2:** A single, concise sentence based **ONLY** on the highest-priority step you identified.
                - **THIS HIERARCHY IS NOT A SUGGESTION; IT IS A STRICT RULE.** Do not describe fallen people (Step 3) if you can see the armed person who caused it (Step 1). Do not describe a crowd (Step 3) if you can see the fight that caused it (Step 2).

                ---
                **Example for your internal logic (User's Image):**
                1.  *Weapon Scan:* Yes, a man is holding a dark object that looks like a firearm.
                2.  *Action:* Stop analysis. My priority is the armed man.
                3.  *Result:*
                    ABNORMAL
                    A person is holding what appears to be a firearm, standing over several people on the floor.

                **Example for a fight:**
                1.  *Weapon Scan:* No.
                2.  *Aggression Scan:* Yes, two people are shoving each other.
                3.  *Action:* Stop analysis. My priority is the fight.
                4.  *Result:*
                    ABNORMAL
                    A physical altercation is occurring between two individuals in the center.

                **Example for panic (no visible cause):**
                1. *Weapon Scan:* No.
                2. *Aggression Scan:* No.
                3. *Event Analysis:* Yes, a crowd is running away in panic.
                4. *Result:*
                    ABNORMAL
                    A large crowd is running away from an unseen threat.
                """
        # --- END OF NEW PROMPT ---

        try:
            img_pil = Image.open(io.BytesIO(frame_bytes))
        except Exception as e:
            self.logger.error(f"Failed to process image bytes: {e}")
            return {"is_abnormal": False, "description": f"Error: Image processing failed ({e})", "timestamp": timestamp}
        
        while retry_count < MAX_API_RETRIES:
            if self.stop_event.is_set():
                return {"is_abnormal": False, "description": "Cancelled", "timestamp": timestamp}
            try:
                response = self.gemini_model.generate_content(
                    [prompt, img_pil], 
                    # --- CRITICAL CHANGE: Set temperature to 0 for maximum rule-following ---
                    generation_config=genai.types.GenerationConfig(temperature=0.0),
                    request_options={'timeout': API_TIMEOUT}
                )
                raw_response = response.text if response.text else "NO_RESPONSE"
                self.logger.debug(f"API Raw Response @{timestamp:.2f}s:\n{raw_response}")
                is_abnormal, description = self._parse_gemini_response(raw_response)
                self.logger.info(f"Parsed @{timestamp:.2f}s: Abnormal={is_abnormal}, Desc='{description}'") # Log full description now
                return {"is_abnormal": is_abnormal, "description": description, "timestamp": timestamp}
            except Exception as e:
                # ... (rest of the error handling is fine) ...
                if "response was blocked" in str(e).lower():
                    self.logger.warning(f"API request potentially blocked (attempt {retry_count+1}).")
                    return {"is_abnormal": True, "description": "Analysis blocked - Potential sensitive content.", "timestamp": timestamp}
                retry_count += 1
                self.logger.warning(f"API request failed (attempt {retry_count}/{MAX_API_RETRIES}): {type(e).__name__}: {e}")
                if retry_count >= MAX_API_RETRIES:
                    self.logger.error(f"API failed after {MAX_API_RETRIES} attempts.")
                    return {"is_abnormal": False, "description": f"Error: API failure ({type(e).__name__})", "timestamp": timestamp}
                delay = API_RETRY_BASE_DELAY * (2 ** (retry_count - 1))
                self.logger.info(f"Retrying in {delay:.2f}s...")
                if self.stop_event.wait(delay):
                    return {"is_abnormal": False, "description": "Cancelled during retry", "timestamp": timestamp}
        
        return {"is_abnormal": False, "description": "Error: Max retries reached", "timestamp": timestamp}
    

    def _analyze_frame_task(self, frame_bytes: bytes, timestamp: float):
        """Background task to analyze a single frame"""
        try:
            result = self._make_api_request(frame_bytes, timestamp)
            self.logger.debug(f"Putting result to queue for T={timestamp:.2f}s: {result}")
            if not self.stop_event.is_set():
                self.results_queue.put(result)
        except Exception as e:
            self.logger.error(f"Exception in analysis task for frame @{timestamp:.2f}s: {e}", exc_info=True)
            if not self.stop_event.is_set():
                self.results_queue.put({"is_abnormal": False, "description": f"Task Error: {e}", "timestamp": timestamp})

    def _draw_text_with_background(self, img, text, origin, font, scale, color, thickness, bg_color=(0, 0, 0), padding=5):
        """Draw text with background for better visibility"""
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        text_w, text_h = text_size
        x, y = origin
        rect_x1 = x - padding
        rect_y1 = y - text_h - padding
        rect_x2 = x + text_w + padding
        rect_y2 = y + padding
        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)
        rect_x2 = min(img.shape[1], rect_x2)
        rect_y2 = min(img.shape[0], rect_y2)
        
        if rect_x1 < rect_x2 and rect_y1 < rect_y2:
            sub_img = img[rect_y1:rect_y2, rect_x1:rect_x2]
            bg_rect = cv2.rectangle(sub_img.copy(), (0, 0), (rect_x2 - rect_x1, rect_y2 - rect_y1), bg_color, cv2.FILLED)
            alpha = 0.6
            res = cv2.addWeighted(bg_rect, alpha, sub_img, 1 - alpha, 1.0)
            img[rect_y1:rect_y2, rect_x1:rect_x2] = res
        
        cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

# In class GeminiVideoAnalyzer:

    def play_and_analyze_video(self, video_path: Optional[str] = None, use_webcam: bool = False):
        """
        Main method to process a video stream, perform analysis, and manage alerts.
        This method features real-time playback synchronization and low-latency audio alerts.

        Args:
            video_path: Path to the video file (if use_webcam is False).
            use_webcam: Boolean flag to use the webcam instead of a file.
        """
        # --- 1. INITIALIZE VIDEO SOURCE ---
        if use_webcam:
            self.logger.info("Opening webcam...")
            cap = cv2.VideoCapture(0)
            source_name = "Webcam"
        elif video_path and os.path.exists(video_path):
            self.logger.info(f"Opening video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            source_name = Path(video_path).name
        else:
            self.logger.error("No valid video source provided.")
            return

        if not cap.isOpened():
            self.logger.error(f"Could not open video source: {source_name}")
            return

        # --- 2. SETUP PLAYBACK TIMING ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            self.logger.warning(f"Source '{source_name}' has invalid FPS. Defaulting to 30.")
            fps = 30
        
        # Calculate the delay in milliseconds required between frames for real-time playback.
        frame_delay_ms = int(1000 / fps)
        self.logger.info(f"Source: {source_name}, Target FPS: {fps:.2f}, Frame Delay: {frame_delay_ms}ms")

        # --- 3. SETUP DISPLAY AND COUNTERS ---
        window_name = "Gemini Violence & Weapon Detector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        frame_count = 0
        start_time = time.time()

        try:
            # Main loop: continues as long as the video source is open and not stopped.
            while cap.isOpened() and not self.stop_event.is_set():
                loop_start_time = time.time()

                # --- 4. READ AND PROCESS A FRAME ---
                ret, frame = cap.read()
                if not ret:
                    if not use_webcam:
                        self.logger.info("End of video file reached.")
                        break  # Exit loop cleanly at the end of a video file.
                    else:
                        self.logger.warning("Webcam returned empty frame. Retrying...")
                        time.sleep(0.1)
                        continue # For webcam, just try again.

                display_frame = frame.copy()
                timestamp = (time.time() - start_time) if use_webcam else (frame_count / fps)

                # Send frame for analysis every Nth frame.
                if frame_count % self.frame_interval == 0:
                    if len(self.futures) < MAX_WORKERS * 2:
                        ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret_encode:
                            future = self.executor.submit(self._analyze_frame_task, buffer.tobytes(), timestamp)
                            self.futures.append(future)
                        else:
                            self.logger.warning(f"Failed to encode frame {frame_count}")
                    else:
                        self.logger.warning("Analysis queue is full, skipping frame to avoid overload.")

                # --- 5. HANDLE API RESULTS & TRIGGER IMMEDIATE AUDIO ALERTS ---
                while not self.results_queue.empty():
                    try:
                        result = self.results_queue.get_nowait()
                        if isinstance(result, dict) and 'is_abnormal' in result:
                            is_abnormal = result.get("is_abnormal", False)
                            description = result.get("description", "No description.")

                            # LOW-LATENCY AUDIO TRIGGER: Fire alert as soon as one abnormal result arrives.
                            # The trigger_alert function has its own cooldown to prevent spamming.
                            if is_abnormal:
                                self.trigger_alert(camera_name=source_name, immediate_description=description)
                            
                            # Update history for VISUAL smoothing and update the last known description.
                            self.classification_history.append(is_abnormal)
                            self.last_description = description
                    except queue.Empty:
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing queue result: {e}")

                # --- 6. TEMPORAL SMOOTHING FOR VISUALS ---
                # This prevents the red border from flickering on single-frame misclassifications.
                if len(self.classification_history) > 0:
                    abnormal_count = sum(1 for is_abn in self.classification_history if is_abn)
                    new_status_is_abnormal = abnormal_count >= ABNORMAL_THRESHOLD_COUNT
                else:
                    new_status_is_abnormal = False
                
                if new_status_is_abnormal and not self.previous_status_is_abnormal:
                    self.logger.info("VISUAL status changed to ABNORMAL due to sustained detection.")
                
                self.current_status_is_abnormal = new_status_is_abnormal
                self.previous_status_is_abnormal = new_status_is_abnormal

                # --- 7. DRAW OVERLAYS ON THE FRAME ---
                status_text = "ABNORMAL" if self.current_status_is_abnormal else "NORMAL"
                status_color = COLOR_ABNORMAL if self.current_status_is_abnormal else COLOR_NORMAL
                
                # Draw main status and timestamp
                self._draw_text_with_background(display_frame, status_text, (20, 40), FONT, FONT_SCALE_STATUS, status_color, FONT_THICKNESS_STATUS)
                self._draw_text_with_background(display_frame, f"T: {timestamp:.2f}s", (20, 80), FONT, FONT_SCALE_INFO, COLOR_INFO, FONT_THICKNESS_INFO)

                # Draw description
                desc_y_start = 120
                wrapped_desc = textwrap.wrap(str(self.last_description), width=DESC_WRAP_WIDTH)
                for i, line in enumerate(wrapped_desc):
                    self._draw_text_with_background(display_frame, line, (20, desc_y_start + i * 25), FONT, FONT_SCALE_INFO, COLOR_INFO, FONT_THICKNESS_INFO)

                # Draw flashing alert banner if an alert needs acknowledgment
                if self.alert_acknowledge_needed:
                    alert_text = "ACTIVE ALERT - PRESS 'C' TO ACKNOWLEDGE"
                    text_size, _ = cv2.getTextSize(alert_text, FONT, FONT_SCALE_STATUS, FONT_THICKNESS_STATUS)
                    text_x = display_frame.shape[1] - text_size[0] - 20
                    flash_on = int(time.time() * 2) % 2 == 0
                    bg_color = (0, 0, 220) if flash_on else (220, 0, 0)
                    self._draw_text_with_background(display_frame, alert_text, (text_x, 50), FONT, FONT_SCALE_STATUS, (255, 255, 255), FONT_THICKNESS_STATUS, bg_color, padding=10)

                # Draw red border if the SMOOTHED visual status is abnormal
                if self.current_status_is_abnormal:
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1] - 1, display_frame.shape[0] - 1), COLOR_ABNORMAL, BORDER_THICKNESS)

                # --- 8. DISPLAY THE FRAME AND HANDLE USER INPUT ---
                cv2.imshow(window_name, display_frame)

                # Calculate wait time to maintain real-time playback speed
                processing_time_ms = (time.time() - loop_start_time) * 1000
                wait_duration = max(1, frame_delay_ms - int(processing_time_ms))
                key = cv2.waitKey(wait_duration) & 0xFF

                if key == ord('q') or key == 27:
                    self.logger.info("Quit key pressed. Shutting down.")
                    break
                elif key == ord('a'):
                    status = "enabled" if self.toggle_alerts() else "disabled"
                    self.logger.info(f"Audio alerts {status}.")
                elif key == ord('c'):
                    if self.acknowledge_alert():
                        self.logger.info("Alert acknowledged by user.")

                # --- 9. LOOP MAINTENANCE ---
                frame_count += 1
                if frame_count % (self.frame_interval * 5) == 0:
                    self._cleanup_futures()

        except KeyboardInterrupt:
            self.logger.info("Keyboard Interrupt detected. Shutting down.")
        finally:
            # --- 10. CLEANUP ALL RESOURCES ---
            self.logger.info("Starting cleanup process...")
            self.stop_event.set()

            self.stop_tts()
            time.sleep(0.2) # Give TTS thread a moment to stop

            # Cancel any tasks still in the pipeline
            cancelled_count = 0
            for future in self.futures:
                if future.cancel():
                    cancelled_count += 1
            if cancelled_count > 0:
                self.logger.info(f"Cancelled {cancelled_count} pending analysis tasks.")
            self.futures.clear()

            # Shut down the thread pool
            self.logger.info("Shutting down worker thread pool...")
            self.executor.shutdown(wait=True)
            self.logger.info("Worker threads shut down.")

            if cap:
                cap.release()
                self.logger.info("Video source released.")

            cv2.destroyAllWindows()
            self.logger.info("Display windows closed.")
            print("Analysis finished.")


# --- Main Function ---
def main():
    load_dotenv()
    print("Attempted to load environment variables from .env file.")
    
    DEFAULT_FRAME_INTERVAL_STR = "30"
    DEFAULT_WEBCAM_MODE_STR = "False"
    DEFAULT_VIDEO_PATH = None
    
    parser = argparse.ArgumentParser(
        description="Analyze video/webcam feed using Google Gemini with Google Cloud TTS.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video_path", nargs='?', default=None, 
                       help="Path to video file. Required if --no-webcam and no hardcoded default.")
    parser.add_argument("--api-key", default=None, 
                       help="Google AI (Gemini) API Key. Overrides GOOGLE_API_KEY env var.")
    
    default_interval = int(os.getenv("DEFAULT_FRAME_INTERVAL", DEFAULT_FRAME_INTERVAL_STR))
    parser.add_argument("--frame-interval", type=int, default=default_interval, 
                       help="Analyze every Nth frame.")
    
    default_webcam = os.getenv("DEFAULT_WEBCAM_MODE", DEFAULT_WEBCAM_MODE_STR).lower() == 'true'
    webcam_group = parser.add_mutually_exclusive_group()
    webcam_group.add_argument("--webcam", action="store_true", 
                             default=default_webcam if default_webcam else None, 
                             help="Use webcam.")
    webcam_group.add_argument("--no-webcam", action="store_false", dest="webcam", 
                             default=argparse.SUPPRESS, help="Force video file input.")
    
    parser.add_argument("--alert-cooldown", type=int, default=ALERT_COOLDOWN, 
                       help=f"Minimum seconds between audio alerts (default: {ALERT_COOLDOWN})")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google AI API Key not found.", file=sys.stderr)
        sys.exit(1)
    
    masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
    print(f"Using API Key: {masked_key}")
    
    # Determine video source
    use_webcam = args.webcam if 'webcam' in args else default_webcam
    video_path = args.video_path
    
    if not use_webcam:
        if video_path is None:
            if DEFAULT_VIDEO_PATH:
                print(f"Using hardcoded default: {DEFAULT_VIDEO_PATH}")
                video_path = DEFAULT_VIDEO_PATH
    elif use_webcam and video_path:
        print("Warning: --webcam used, ignoring video path.")
        video_path = None
    
    # Validate input source
    if use_webcam:
        pass  # Webcam validation happens in the analyzer
    elif not video_path:
        print("Error: No input source specified.", file=sys.stderr)
        sys.exit(1)
    elif not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Validate parameters
    if args.frame_interval <= 0:
        print(f"Error: --frame-interval must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.alert_cooldown < 0:
        print(f"Error: --alert-cooldown cannot be negative.", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration
    print("\n--- Starting Gemini Video Analyzer w/ Google Cloud TTS ---")
    print(f"Mode: {'Webcam' if use_webcam else f'Video File ({video_path})'}")
    print(f"Analysis Frame Interval: {args.frame_interval}")
    print(f"Audio Alert Cooldown: {args.alert_cooldown}s")
    print("Press 'q' or ESC to quit.")
    print("Press 'a' to toggle audio alerts.")
    print("Press 'c' to acknowledge active alerts.")
    print("--------------------------------------\n")
    
    try:
        analyzer = GeminiVideoAnalyzer(
            api_key=api_key, 
            frame_interval=args.frame_interval, 
            alert_cooldown=args.alert_cooldown
        )
        analyzer.play_and_analyze_video(video_path=video_path, use_webcam=use_webcam)
    except ValueError as ve:
        print(f"\nConfiguration or Value Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Exiting gracefully.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        logging.exception("Unexpected error details:")
        sys.exit(1)
    finally:
        print("\n--- Analysis Session Ended ---")


if __name__ == "__main__":
    main()