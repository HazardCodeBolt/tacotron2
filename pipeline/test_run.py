from pipeline import VoiceDatasetPipeline
import os

def test_pipeline():
    # Note: We use a small max_videos count for testing.
    # To run this, ensure you have internet access and the profile is public.
    pipeline = VoiceDatasetPipeline()
    
    # We will use 'instagram' as a test account which is public and has videos.
    # In a real use case, the user would replace this with their target account.
    try:
        pipeline.run("instagram", max_videos=1)
        print("Test run completed successfully.")
    except Exception as e:
        print(f"Test run failed: {e}")

if __name__ == "__main__":
    # For the purpose of this environment, we won't run the full download
    # but we will check if the modules load correctly.
    print("Checking pipeline modules...")
    try:
        import downloader
        import audio_processor
        import speaker_manager
        import cleaner
        import annotator
        print("All modules loaded successfully.")
    except ImportError as e:
        print(f"Module load error: {e}")
