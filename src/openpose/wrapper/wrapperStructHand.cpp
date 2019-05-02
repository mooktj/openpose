#include <openpose/wrapper/wrapperStructHand.hpp>
#include <iostream>

namespace op
{
    WrapperStructHand::WrapperStructHand(
        const bool enable_, const Detector detector_, const Point<int>& netInputSize_, const int scalesNumber_,
        const float scaleRange_, const RenderMode renderMode_, const float alphaKeypoint_, const float alphaHeatMap_,
        const float renderThreshold_) :
        enable{enable_},
        detector{detector_},
        netInputSize{netInputSize_},
        scalesNumber{scalesNumber_},
        scaleRange{scaleRange_},
        renderMode{renderMode_},
        alphaKeypoint{alphaKeypoint_},
        alphaHeatMap{alphaHeatMap_},
        renderThreshold{renderThreshold_}
    {
        // std::cout << "wrapperStructHand:: WrapperStructHand(...) constructor\n";
    }
}
