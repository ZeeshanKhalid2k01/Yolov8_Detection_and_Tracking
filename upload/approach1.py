def approach(x,video_path):

    import time
    import cv2
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    # Load the YOLOv8 model
    if x=='n':
        model = YOLO('../yolov8n-pose.pt')
    else:
        model = YOLO('../yolov8s-pose.pt')

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    fps_values = []  # List to store FPS values
    total_fps = 0  # Total FPS for the video

    # Loop through the frames
    while cap.isOpened():
        # Read a frame from the camera
        success, frame = cap.read()

        if success:
            start = time.perf_counter()

            # Run YOLOv8 inference on the frame
            results = model(frame)

            end = time.perf_counter()
            total_time = end - start
            fps = 1 / total_time

            # Store FPS value
            fps_values.append(fps)

            # Visualize the results
            annotated_frame = results[0].plot()

            # Display the annotated frame
            text = f"FPS: {int(fps)}"
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLOv8", annotated_frame)

            # Exit if 'q' is pressed or if the video ends
            key = cv2.waitKey(1)
            if key == ord('q') or not success or len(fps_values) == 300:
                break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

    # Display FPS graph
    plt.plot(fps_values)
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.title("FPS Over Time")
    if(x=='s'):
        plt.savefig('Approach1_fps_graph_v8s.png')
    else:
        plt.savefig('Approach1_fps_graph_v8n.png')
    # plt.show()

    # Calculating the average FPS
    for fps in fps_values:
        total_fps += fps

    avg_fps = total_fps / len(fps_values)   
    max_fps = max(fps_values)
    min_fps = min(fps_values)
    # print(f"Average FPS: {avg_fps}")
    # print(f"Max FPS are: ",{max_fps})
    # print(f"Min FPS are: ",{min_fps})
    # print("AVG FPS, MAX FPS, MIN FPS")

    return(avg_fps,max_fps,min_fps)