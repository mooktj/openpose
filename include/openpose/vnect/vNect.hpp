#ifndef OPENPOSE_VNECT_HPP
#define OPENPOSE_VNECT_HPP

#include <openpose/core/common.hpp>

namespace op
{
    float vNectFindMax(std::vector<float> in);
    float vNectFindMin(std::vector<float> in);

    void vNectPostForward(const std::shared_ptr<op::Datum>& datumsPtr);
    std::vector<std::vector<float>> vNectForward(cv::Mat croppedImg);
}

#endif // OPENPOSE_VNECT_HPP