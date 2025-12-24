# import cv2

# from ultralytics import YOLO, solutions

# model = YOLO("yolo11n-pose.pt")
# cap = cv2.VideoCapture("pull.mp4")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# video_writer = cv2.VideoWriter("pull_up_mousa.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# gym_object = solutions.AIGym(
#     line_thickness=2,
#     view_img=True,
#     pose_type="pullup",
#     kpts_to_check=[5, 7, 9]
# )
# frame_count=0
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     frame_count+=1
#     results = model.track(im0, verbose=False)  # Tracking recommended
#     #results = model.predict(im0)  # Prediction also supported
#     im0 = gym_object.start_counting(im0, results)
#     video_writer.write(im0)

# cv2.destroyAllWindows()
# video_writer.release()


#---------------------------------------

import cv2
from ultralytics import solutions,YOLO
#model=YOLO("yolo11s-pose.pt")
# model=YOLO("yolo11n.pt")
cap = cv2.VideoCapture("lego.mp4")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = cv2.VideoWriter("leg.mp4",fourcc,30,(720,1280))

# Init AIGym
gym = solutions.AIGym(
    kpts=[6, 12, 14],  
    model="yolo11s-pose.pt",
    conf=0.8
    # up_angle=120.0,
    # down_angle = 80.0

)

counter=0
stage=''
# Process video
while True:

    ret, frame = cap.read()
    if not ret:
        break
    
    #show = cv2.resize(frame,(480,640))  
    frame = cv2.resize(frame,(720,1280)) 
    show = frame.copy() 
    gym.monitor(frame)

    if gym.angle[0]>=120:
        stage='up'
    if gym.angle[0]<=80 and stage=='up':
        stage='down'
        counter+=1

    
    cv2.putText(show,'Counter= '+str(counter),(5,50),cv2.FONT_HERSHEY_COMPLEX,(2),(0,0,255),3)


    video_writer.write(frame)
    cv2.imshow('gym_counter',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
#video_writer.release()