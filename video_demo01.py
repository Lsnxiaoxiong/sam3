
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(checkpoint_path=r"C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3\sam3.pt")
video_path = "video" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="keyboard",
    )
)
output = response["outputs"]