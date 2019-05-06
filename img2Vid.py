import cv2


file_image = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-6-5-19/drawVNect/90-80/frame_0.jpg"
img = cv2.imread(file_image)
print "img shape: ", img.shape # (h, w, c)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('test_output.avi', fourcc, 10.0, (img.shape[1], img.shape[0]))

for file_i in range(0,150):
	print "processing frame ", file_i
	file_image = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-6-5-19/drawVNect/90-80/frame_" + str(file_i) + ".jpg"
	img = cv2.imread(file_image)
	out.write(img)

out.release()
