#ifndef OPENPOSE_VNECT_SNOWMAN_HPP
#define OPENPOSE_VNECT_SNOWMAN_HPP

#include <openpose/core/common.hpp>

namespace op
{
	cv::Point getFloorLevelPt(std::vector<float> snowMan);
	float inferDepth(float xlbot_a, float ylbot_a, float xlbot_b, float ylbot_b, cv::Point vanishPt);
	float heightInit(std::vector<float> snowman, cv::Point vanishPt, cv::Point groundPlane);
	float compareSnowmenSize(std::vector<float> currSnow, std::vector<float> prevSnow);
    
	float reMatchSnowmen(std::vector<float> prev, std::vector<float> curr);
	std::vector<float> updateSnowman(std::vector<float> prevSnowman, std::vector<float> currSnowman, cv::Point groundPlane);
    std::vector<float> reAdjustSnowman(std::vector<float> prevSnowman, std::vector<float> currSnowman);
    bool checkFullSnowman(std::vector<float> currSnowman);
}

#endif // OPENPOSE_VNECT_SNOWMAN_HPP