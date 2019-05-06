import cv2

video = cv2.VideoCapture("/home/mooktj/Desktop/myworkspace/Gogobebe_dance_cover.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
print "fps of video is ", int(fps)

video.release() 