# Video Feature Extraction Tool

This is a small Python script I built to analyze a local video file and extract a few high-level visual and temporal features from it.

## What I Extracted

From the input video, I extracted the following features:

### Hard Cut Count
I detected hard cuts by comparing the mean pixel intensity between consecutive frames.
If the difference crosses a threshold, I count it as a scene cut.

### Average Motion (Optical Flow)
I used dense optical flow (Farneback) to measure motion between consecutive frames.

For each frame pair, I computed the optical flow magnitude and then averaged it across the entire video to get a single motion value.

### Text Present Ratio
I used Tesseract OCR only to check whether any text appears in a frame and determine a text_present_ratio

The final value is calculated as:
frames_with_text / total_frames

### Person vs Object Ratio
I used a pre-trained YOLOv8 model to detect objects in each frame.

All detections labeled person are counted as people

All other detected classes are counted as objects

The ratio gives a simple indication of whether the video is more person-focused or object-focused.

### Example Output
{
  'hard_cut_count': 4,
  'average_motion': 1.20,
  'text_present_ratio': 0.99,
  'person_object_ratio': 14.8
}

### How to Run
-Clone the repository

-Install the dependencies pip install -r requirements.txt

-add the filepath to your video file in the filepath variable at the top

-Run python app.py

### Notes

The focus was on feature extraction, not perfect detection accuracy

OCR was intentionally limited to text presence detection

Person/object counts are aggregated across frames without identity tracking

### Author

Daniel Sholademi