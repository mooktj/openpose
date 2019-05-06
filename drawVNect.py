import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

# file_path = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-3-5-19/joints_3d.txt"

ims = []

fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
ax = Axes3D(fig)
ax.clear()

ax.view_init(azim=-60, elev=-70)


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

for file_i in range(0, 150):

	print "processing frame ", file_i

	file_path = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-5-5-19/Gogobebejoints_3d" + str(file_i) + ".txt"

	poses = []
	poseCount = 0
	currPose = []
	avgDiffs = []
	floorLevels = []

	with open(file_path) as file:
		for line in file:
			if len(line) == 1: continue
			if "pose_" in line.split()[0]: 
				index = line.split()[0].find('_')
				# print line.split()[0][index+1]
				if poseCount != int(line.split()[0][index+1]):
					# print "TO APPEND NEW POSE ARRAY"
					poses.append(currPose)
					currPose = []
					poseCount = int(line.split()[0][index+1])
				currPose.append(float(line.split()[1]))
				currPose.append(float(line.split()[2]))
				currPose.append(float(line.split()[3]))

			if "POSE_END" in line: 
				# print "FOUND POSE_END"
				poses.append(currPose)

			if "avgDiffs" in line.split():
				# print line
				avgDiffs.append(float(line.split()[1]))

			if "floorLevels" in line.split():
				floorLevels.append(float(line.split()[1]))


	# for i in range(len(poses)):
		# print "   ////// i = ", i
		# for j in range(0, len(poses[i]), 3):
			# print poses[i][j], " ", poses[i][j+1], " ", poses[i][j+2]

	# print "print avgDiffs"
	# for i in range(len(avgDiffs)):
		# print avgDiffs[i]

	# print "print floorLevels"
	# for i in range(len(floorLevels)):
		# print floorLevels[i]

	limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]


	fig = plt.figure()
	# ax = fig.add_subplot(121, projection='3d')
	ax = Axes3D(fig)
	ax.clear()

	ax.view_init(azim=-90, elev=-80)


	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	# print "\n\n"


	for i in range(len(poses)):
		for j in range(0, len(poses[i]), 3):
			# print "j: ", j
			if i == 0:
				x_pair = [poses[i][(j)]	, poses[i][3 * limb_parents[(j/3)]]]
				y_pair = [poses[i][(j)+1]	, poses[i][(3 * limb_parents[(j/3)]) + 1]]
				z_pair = [poses[i][(j)+2]	, poses[i][(3 * limb_parents[(j/3)]) + 2]]
				ax.plot(x_pair, y_pair,zs=z_pair,linewidth=3)
			else:
				offset_x = avgDiffs[i-1]
				offset_z = floorLevels[i-1]
				# print "-----------------IN PYTHON----------------"
				# print "offset_x: ", offset_x
				# print "offset_z: ", offset_z
				# print "-----------------PRINTED PYTHON----------------"
				x_pair = [poses[i][(j)] + offset_x	, poses[i][3 * limb_parents[(j/3)]] + offset_x]
				y_pair = [poses[i][(j)+1]				, poses[i][(3 * limb_parents[(j/3)]) + 1]]
				z_pair = [poses[i][(j)+2] + offset_z	, poses[i][(3 * limb_parents[(j/3)]) + 2] + offset_z]
				ax.plot(x_pair, y_pair,zs=z_pair,linewidth=3)

	fig.savefig("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-6-5-19/drawVNect/90-80/" + "frame_" + str(file_i) + ".jpg")
	# ims.append(ax)
	# plt.show()

# im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
# # im_ani.save("im.mp4", writer=writer)
# im_ani.save("im.avi", codec='avi')