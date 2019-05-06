# Program To Read video 
# and Extract Frames 
import cv2 

# Function to extract frames 
def FrameCapture(path): 
	
	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	while success: 
		print "creating frame ", count
		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Saves the frames with frame-count 
		cv2.imwrite("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_%d.jpg" % count, image) 

		count += 1

# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	# FrameCapture("/home/mooktj/Desktop/myworkspace/img2vid/kim_yuna/Kim Yuna 2013 World Championship Free Skating Practice.mp4") 
	FrameCapture("/home/mooktj/Desktop/myworkspace/Gogobebe_dance_cover.mp4") 
	